import logging

import torch
from torch.nn.utils import parameters_to_vector
import copy,math

from collections import defaultdict
import numpy as np
import sklearn.metrics.pairwise as smp
from decentralizepy.sharing.Sharing import Sharing


class PlainAverageSharing(Sharing):
    """
    Class to do plain averaging instead of Metropolis Hastings
    """

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        compress=False,
        compression_package=None,
        compression_class=None,
        float_precision=None,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank
        machine_id : int
            Global machine id
        communication : decentralizepy.communication.Communication
            Communication module used to send and receive messages
        mapping : decentralizepy.mappings.Mapping
            Mapping (rank, machine_id) -> uid
        graph : decentralizepy.graphs.Graph
            Graph reprensenting neighbors
        model : decentralizepy.models.Model
            Model to train
        dataset : decentralizepy.datasets.Dataset
            Dataset for sharing data.
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)

        """
        super().__init__(
            rank,
            machine_id,
            communication,
            mapping,
            graph,
            model,
            dataset,
            log_dir,
            compress,
            compression_package,
            compression_class,
            float_precision,
        )
        self.received_this_round = 0

    def _pre_step(self):
        """
        Called at the beginning of step.

        """
        pass

    def _post_step(self):
        """
        Called at the end of step.

        """
        pass
    
    def distance_calculate(self,k,b,center):
        """
        计算点到直线的距离
        
        """
        w=center-b
        alpha=torch.dot(k,w)/torch.dot(k,k)
        if alpha>=0:
            pt=b+alpha*k
            dis=torch.norm(pt-center)
        else:
            dis=torch.norm(w)
            pt=b
        return dis,pt

    def superball_calculate(self, model_history,iteration):
        """
        模拟退火求覆盖射线集的超球
        
        """
        if(len(model_history)<=1):
            return -1,-1
        T = 30
        tao = 100 #10000
        TAO_0 = 1e-6
        ALPHA = 0.98
        ZETA = 0.9
        initial_center = torch.zeros_like(model_history[1])
        num = 0
        for i in range(max(1,iteration-T+1)-1,iteration):
            initial_center += model_history[i]
            num += 1
        center = initial_center/num
        dis_list = list()
        # logging.info("max(1,iteration-T+1):{},iteration:{}".format(max(1,iteration-T+1),iteration))
        for i in range(max(1,iteration-T+1)-1,iteration):
            k=model_history[i+1]-model_history[i]
            b=model_history[i]
            dis,pt=self.distance_calculate(k,b,center)
            # logging.info("---dis:{},pt:{},center:{}".format(dis,pt,center))
            dis_list.append((dis,pt))
        dis_list.sort(key=lambda x:x[0].item())
        # logging.info("len(dis_list):{},int(len(dis_list)*ZETA):{}".format(len(dis_list),int(len(dis_list)*ZETA)))
        radius = dis_list[int(len(dis_list)*ZETA)][0]
        while tao > TAO_0:
            # logging.info("---center:{},radius:{}".format(center,radius))
            mpt=dis_list[len(dis_list)-1][1]
            acenter = center + tao*((mpt-center)/torch.norm(mpt-center))
            dis_list = list()
            for i in range(max(1,iteration-T+1)-1,iteration):
                k=model_history[i+1]-model_history[i]
                b=model_history[i]
                dis,pt=self.distance_calculate(k,b,acenter)
                dis_list.append((dis,pt))
            dis_list.sort(key=lambda x:x[0].item())   
            aradius = dis_list[int(len(dis_list)*ZETA)][0]
            if aradius < radius:
                center = acenter
                radius = aradius
            else:
                p = torch.exp((radius-aradius)/tao)
                if torch.rand(1).item() < p.item():
                    center = acenter
                    radius = aradius
            tao *= ALPHA
        return center,radius
    
    def calculate_similarity(self,center,centerx,model_history,iteration):
        T = 30
        GAMA = 0.9

        g_i=torch.zeros_like(model_history[1])
        for i in range(max(1,iteration-T+1)-1,iteration):
            g_i += math.exp(-GAMA*(iteration-1-i))*(model_history[i+1]-model_history[i])
        
        O_ik = centerx - center
        if torch.dot(g_i,O_ik) > 0:
            Sim = 1
        elif torch.abs(torch.dot(g_i,O_ik)) < 1e-8:
            Sim = 0
        else:
            Sim = -1
        
        Sim *= 1/(1+O_ik.norm().item())
        return Sim
    
    def rep_evaluation(self,Sim_x,radius,radiusx,max_radius,min_radius):
        B = 0.5
        C = 0.5

        # logging.info("{},{},{},{},{},{}".format(type(Sim_x),Sim_x,radius,radiusx,max_radius,min_radius))
        pi = (Sim_x + 1)/2
        ni = (1 - Sim_x)/2
        ux = 1/(1+torch.exp(-(radius+radiusx-2*min_radius)/(max_radius-min_radius+1e-8)).item())
        bx = (1-ux)*(B*pi/(B*pi+C*ni))
        dx = (1-ux)*(C*ni/(B*pi+C*ni))
        # logging.info("ux:{},bx:{},dx:{}".format(ux,bx,dx))
        return bx,dx,ux
    
    def _averaging(self, peer_deques,global_lr,model_history,iteration,my_neighbors):
        """
        Averages the received model with the local model

        """
        A = 0.5 
        centers,radiuss = dict(),dict()
        with torch.no_grad():
            for x in model_history:
                # logging.info("iteration:{},x:{},model_history:{}".format(iteration,x,model_history))
                n_model=copy.deepcopy(self.model)
                n_model.load_state_dict(self.deserialized_model(model_history[x][iteration]))
                model_history[x][iteration] = parameters_to_vector(n_model.parameters()).to("cuda")
                logging.info("model_history[{}]:{}".format(x,model_history[x]))
        
            center,radius=self.superball_calculate(model_history[self.rank],iteration)
            logging.info("me:{},center:{},radius:{}".format(self.rank,center,radius))
            centers[self.rank] = center
            radiuss[self.rank] = radius
            for x in my_neighbors:
                centerx,radiusx=self.superball_calculate(model_history[x],iteration)
                logging.info("x:{},center:{},radius:{}".format(x,centerx,radiusx))
                centers[x] = centerx
                radiuss[x] = radiusx

            reps = dict()
            for x in my_neighbors:
                if type(centers[x]) is int and centers[x] == -1 and radiuss[x] == -1:
                    reps[x] = -1
                else:
                    # logging.info("centers:{},radiuss:{}".format(centers,radiuss))
                    Sim_x = self.calculate_similarity(centers[self.rank],centers[x],model_history[self.rank],iteration)
                    stacked_radius_tensors=torch.stack(list(radiuss.values()))
                    max_radius=torch.max(stacked_radius_tensors)
                    min_radius=torch.min(stacked_radius_tensors)
                    logging.info("max_radius:{},min_radius:{}".format(max_radius,min_radius))
                    bx,dx,ux=self.rep_evaluation(Sim_x,radiuss[self.rank],radiuss[x],max_radius,min_radius)
                    repx=bx+A*ux
                    reps[x] = repx
                    logging.info("Sim_x:{}".format(Sim_x))
                logging.info("reps[x]:{},{}".format(reps[x],type(reps[x])))

        ITA = 0.8
        self.received_this_round = 0
        with torch.no_grad():
            Agg = dict()
            Agg[self.rank] = 1
            for x in reps:
                if reps[x] > ITA:
                    Agg[x] = reps[x]
            
            total = dict()
            sums = sum(Agg.values())
            train_data = dict()
            for i, n in enumerate(peer_deques):
                self.received_this_round += 1
                data = peer_deques[n].popleft()
                iteration = data["iteration"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                if n in Agg:
                    data = self.deserialized_model(data)
                    for key, value in data.items():
                        if key in total:
                            total[key] += value * Agg[n]/sums
                        else:
                            total[key] = value * Agg[n]/sums
                else:
                    logging.info("Neighbor {} is not in Agg".format(n))

            for key, value in self.model.state_dict().items():
                if key in total.keys():
                    total[key] += value * Agg[self.rank]/sums
                else:
                    total[key] = value * Agg[self.rank]/sums

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1

    def get_data_to_send(self, *args, **kwargs):
        self._pre_step()
        data = self.serialized_model()
        data["iteration"] = self.communication_round
        return data

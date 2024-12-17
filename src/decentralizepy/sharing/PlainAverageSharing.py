import logging

import torch
from torch import parameters_to_vector

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
        tao = 100000
        TAO_0 = 3e-7
        ALPHA = 0.999
        ZETA = 0.9
        initial_center = torch.zeros_like(model_history[1])
        num = 0
        for i in range(max(1,iteration-T+1),iteration):
            initial_center += model_history[i]
            num += 1
        center = initial_center/num
        dis_list = list()
        for i in range(max(1,iteration-T+1),iteration):
            k=model_history[i+1]-model_history[i]
            b=model_history[i]
            dis,pt=self.distance_calculate(self,k,b,center)
            dis_list.append((dis,pt))
        dis_list.sort(key=lambda x:x[0])
        radius = dis_list[int(len(dis_list)*ZETA)]
        while tao > TAO_0:
            mpt=dis_list[len(dis_list)-1][1]
            acenter = center + tao*((mpt-center)/torch.norm(mpt-center))
            dis_list = list()
            for i in range(max(1,iteration-T+1),iteration):
                k=model_history[i+1]-model_history[i]
                b=model_history[i]
                dis,pt=self.distance_calculate(self,k,b,acenter)
                dis_list.append((dis,pt))
            dis_list.sort(key=lambda x:x[0])   
            aradius = dis_list[int(len(dis_list)*ZETA)]
            if aradius < radius:
                center = acenter
                radius = aradius
            else:
                p = np.exp((radius-aradius)/tao)
                if np.random.rand() < p:
                    center = acenter
                    radius = aradius
            tao *= ALPHA
        return center,radius
    
    def calculate_similarity(self,center,centerx,model_history,iteration):
        T = 30
        GAMA = 0.9

        g_i=np.zeros_like(model_history[1])
        for i in range(max(1,iteration-T+1),iteration):
            g_i += np.exp(-GAMA*(iteration-1-i))*(model_history[i+1]-model_history[i])
        
        O_ik = centerx - center
        if np.dot(g_i,O_ik) > 0:
            Sim = 1
        elif np.dot(g_i,O_ik) == 0:
            Sim = 0
        else:
            Sim = -1
        
        Sim *= 1/(1+O_ik.norm())
        return Sim
    
    def rep_evaluation(self,Sim_x,radius,radiusx,max_radius,min_radius):
        B = 0.5
        C = 0.5

        pi = (Sim_x + 1)/2
        ni = (1 - Sim_x)/2
        ux = 1/(1+np.exp(-(radius+radiusx-2*min_radius)/(max_radius-min_radius)))
        bx = (1-ux)*(B*pi/(B*pi+C*ni))
        dx = (1-ux)*(C*ni/(B*pi+C*ni))
        return bx,dx,ux
    
    def _averaging(self, peer_deques,global_lr,model_history,iteration,my_neighbors):
        """
        Averages the received model with the local model

        """
        A = 0.5 
        centers,radiuss = dict(),dict()
        for x in model_history:
            model_history[x][iteration] = parameters_to_vector(self.deserialized_model(model_history[x][iteration])) 
        with torch.no_grad():
            center,radius=self.superball_calculate(self,model_history[self.machine_id],iteration)
            centers[self.machine_id] = center
            radiuss[self.machine_id] = radius
            for x in my_neighbors:
                centerx,radiusx=self.superball_calculate(self,model_history[x],iteration)
                centers[x] = centerx
                radiuss[x] = radiusx

            reps = dict()
            for x in my_neighbors:
                Sim_x = self.calculate_similarity(self,centers[self.machine_id],centers[x],model_history[self.machine_id],iteration)
                bx,dx,ux=self.rep_evaluation(self,Sim_x,radiuss[self.machine_id],radiuss[x],max(radiuss.values()),min(radiuss.values()))
                repx=bx+A*ux
                reps[x] = repx

        ITA = 0.8
        self.received_this_round = 0
        with torch.no_grad():
            Agg = dict()
            Agg[self.machine_id] = 1
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
                total[key] += value * Agg[self.machine_id]/sums

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1

    def get_data_to_send(self, *args, **kwargs):
        self._pre_step()
        data = self.serialized_model()
        data["iteration"] = self.communication_round
        return data

import logging

import torch

from collections import defaultdict
import numpy as np
import math
import hdbscan
import sklearn.metrics.pairwise as smp
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
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

    def flame(self,trained_params, current_model_param, param_updates,participant_sample_size):
        # === clustering ===
        trained_params = torch.stack(trained_params).double()
        cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic",
                                min_cluster_size=participant_sample_size // 2 + 1,
                                min_samples=1, allow_single_cluster=True)
        cluster.fit(trained_params)
        predict_good = []
        for i, j in enumerate(cluster.labels_):
            if j == 0:
                predict_good.append(i)
        k = len(predict_good)

        # === median clipping ===
        model_updates = trained_params[predict_good] - current_model_param
        local_norms = torch.norm(model_updates, dim=1)
        S_t = torch.median(local_norms)
        scale = S_t / local_norms
        scale = torch.where(scale > 1, torch.ones_like(scale), scale)
        model_updates = model_updates * scale.view(-1, 1)

        # === aggregating ===
        trained_params = current_model_param + model_updates
        trained_params = trained_params.sum(dim=0) / k

        # === noising ===
        delta = 1 / (participant_sample_size ** 2)
        epsilon = 10000
        lambda_ = 1 / epsilon * (math.sqrt(2 * math.log((1.25 / delta))))
        sigma = lambda_ * S_t.numpy()
        print(f"sigma: {sigma}; #clean models / clean models: {k} / {predict_good}, median norm: {S_t},")
        trained_params.add_(torch.normal(0, sigma, size=trained_params.size()))

        # === bn ===
        global_update = dict()
        for i, (name, param) in enumerate(param_updates.items()):
            if 'num_batches_tracked' in name:
                global_update[name] = 1 / k * \
                                    param_updates[name][predict_good].sum(dim=0, keepdim=True)
            elif 'running_mean' in name or 'running_var' in name:
                local_norms = torch.norm(param_updates[name][predict_good], dim=1)
                S_t = torch.median(local_norms)
                scale = S_t / local_norms
                scale = torch.where(scale > 1, torch.ones_like(scale), scale)
                global_update[name] = param_updates[name][predict_good] * scale.view(-1, 1)
                global_update[name] = 1 / k * global_update[name].sum(dim=0, keepdim=True)

        return trained_params.float().to("cpu"), global_update

    
    def _averaging(self, peer_deques,global_lr,participant_sample_size):
        """
        Averages the received model with the local model

        """
        self.received_this_round = 0
        with torch.no_grad():
            total = dict()
            train_params = dict()
            weight = 1 / (len(peer_deques) + 1)
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
                data = self.deserialized_model(data)
                cli_model = copy.deepcopy(self.model)
                cli_model.load_state_dict(data)
                train_param = parameters_to_vector(cli_model.parameters()).detach().cpu()
                train_params.append(train_param)

                for name, param in data.items():
                    if name not in train_data:
                        train_data[name] = param.data.view(1, -1)
                    else:
                        train_data[name] = torch.cat((train_data[name], param.data.view(1, -1)),
                                                        dim=0)
                
            model_updates = dict()
            for (name, param), local_param in zip(self.model.state_dict().items(), train_data.values()):
                model_updates[name] = local_param.data - param.data.view(1, -1)
            
            current_model_param = parameters_to_vector(self.model.parameters()).detach().cpu()
            global_param, global_update = self.flame(train_params, current_model_param, model_updates,participant_sample_size)
            vector_to_parameters(global_param, self.model.parameters())
            model_param = self.model.state_dict()
            for name, param in model_param.items():
                if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
                    model_param[name] = param.data + global_update[name].view(param.size())
            #     for key, value in data.items():
            #         if key in total:
            #             total[key] += value * weight
            #         else:
            #             total[key] = value * weight

            # for key, value in self.model.state_dict().items():
            #     total[key] += value * weight

        self.model.load_state_dict(model_param)
        self._post_step()
        self.communication_round += 1

    def get_data_to_send(self, *args, **kwargs):
        self._pre_step()
        data = self.serialized_model()
        data["iteration"] = self.communication_round
        return data

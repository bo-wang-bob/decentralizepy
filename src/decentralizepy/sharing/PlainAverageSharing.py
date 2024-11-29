import logging

import torch

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

    def foolsgold(self,model_updates):
        keys = list(model_updates.keys())
        last_layer_updates = model_updates[keys[-2]]
        K = len(last_layer_updates)
        cs = smp.cosine_similarity(last_layer_updates.cpu().numpy()) - np.eye(K)
        maxcs = np.max(cs, axis=1)
        # === pardoning ===
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        alpha = np.max(cs, axis=1)
        wv = 1 - alpha
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # === Rescale so that max value is wv ===
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # === Logit function ===
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        # === calculate global update ===
        global_update = defaultdict()
        for name in keys:
            tmp = None
            for i, j in enumerate(range(len(wv))):
                if i == 0:
                    tmp = model_updates[name][j] * wv[j]
                else:
                    tmp += model_updates[name][j] * wv[j]
            global_update[name] = 1 / len(wv) * tmp

        return global_update
    
    def _averaging(self, peer_deques,global_lr):
        """
        Averages the received model with the local model

        """
        self.received_this_round = 0
        with torch.no_grad():
            total = dict()
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
                for name, param in data.items():
                    if name not in train_data:
                        train_data[name] = param.data.view(1, -1)
                    else:
                        train_data[name] = torch.cat((train_data[name], param.data.view(1, -1)),
                                                        dim=0)
                
            model_updates = dict()
            for (name, param), local_param in zip(self.model.state_dict().items(), train_data.values()):
                model_updates[name] = local_param.data - param.data.view(1, -1)
            
            global_update = self.foolsgold(model_updates)
            
            for name, param in self.model.state_dict().items():
                total[name] = param.data + global_lr * global_update[name].view(param.data.shape)
            #     for key, value in data.items():
            #         if key in total:
            #             total[key] += value * weight
            #         else:
            #             total[key] = value * weight

            # for key, value in self.model.state_dict().items():
            #     total[key] += value * weight

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1

    def get_data_to_send(self, *args, **kwargs):
        self._pre_step()
        data = self.serialized_model()
        data["iteration"] = self.communication_round
        return data

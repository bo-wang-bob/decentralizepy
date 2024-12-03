import logging

import torch

from collections import defaultdict
import numpy as np
import sklearn.metrics.pairwise as smp
import copy
import torch.nn.functional as F
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

    def fltrust(self,model_updates, param_updates, clean_param_update):
        cos = torch.nn.CosineSimilarity(dim=0)
        g0_norm = torch.norm(clean_param_update)
        weights = []
        for param_update in param_updates:
            weights.append(F.relu(cos(param_update.view(-1, 1), clean_param_update.view(-1, 1))))
        weights = torch.tensor(weights).to("cpu").view(1, -1)
        weights = weights / weights.sum()
        weights = torch.where(weights[0].isnan(), torch.zeros_like(weights), weights)
        nonzero_weights = torch.count_nonzero(weights.flatten())
        nonzero_indices = torch.nonzero(weights.flatten()).flatten()

        # print(f'g0_norm: {g0_norm}, '
        #     f'weights_sum: {weights.sum()}, '
        #     f'*** {nonzero_weights} *** model updates are considered to be aggregated !')

        normalize_weights = []
        for param_update in param_updates:
            normalize_weights.append(g0_norm / torch.norm(param_update))

        global_update = dict()
        for name, params in model_updates.items():
            if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
                global_update[name] = 1 / nonzero_weights * params[nonzero_indices].sum(dim=0, keepdim=True)
            else:
                global_update[name] = torch.matmul(
                    weights,
                    params * torch.tensor(normalize_weights).to("cpu").view(-1, 1))
        return global_update
    
    def _averaging(self, peer_deques,global_lr,iterations):
        """
        Averages the received model with the local model

        """
        self.received_this_round = 0
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            total = dict()
            weight = 1 / (len(peer_deques) + 1)
            train_data = dict()
            param_updates = list()
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
                trained_local_model = copy.deepcopy(self.model) 
                trained_local_model.load_state_dict(data)
                param_updates.append(parameters_to_vector(trained_local_model.parameters()) - parameters_to_vector(self.model.parameters()))
                for name, param in data.items():
                    if name not in train_data:
                        train_data[name] = param.data.view(1, -1)
                    else:
                        train_data[name] = torch.cat((train_data[name], param.data.view(1, -1)),
                                                        dim=0)
                
            model_updates = dict()
            
            for (name, param), local_param in zip(self.model.state_dict().items(), train_data.values()):
                model_updates[name] = local_param.data - param.data.view(1, -1)

        if iterations > 500:
            lr = global_lr * 0.991 ** ((iterations - 500) // 5)
        else:
            lr = global_lr
        model = copy.deepcopy(self.model)
        model.load_state_dict(self.model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        model = model.to(device)
        epochs = 2
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            for inputs, labels in self.dataset.get_trainset():
                inputs, labels = inputs.to(device), labels.to(device)
                # print(inputs, labels)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

        clean_param_update = parameters_to_vector(model.to("cpu").parameters()) - parameters_to_vector(
            self.model.parameters())

        global_update = self.fltrust(model_updates, param_updates, clean_param_update)
        
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

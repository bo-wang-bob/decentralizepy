import logging

import torch
from torch.nn.utils import parameters_to_vector
import copy, math

from collections import defaultdict
import numpy as np
import sklearn.metrics.pairwise as smp
from decentralizepy.sharing.Sharing import Sharing
from decentralizepy import utils
import hdbscan
from sklearn.cluster import DBSCAN, KMeans


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


    def _averaging_by_shared_tensor_with_avg(
        self, shared_tensor, current_idx, procs_per_machine, T
    ):
        device = torch.device("cpu")
        latest_model1_index = (current_idx - 2 + 2 * T) % (2 * T)  # 第0个节点
        # 找到所有的中间模型
        model2s = []
        for _ in range(procs_per_machine):
            model2s.append(shared_tensor[latest_model1_index + 1].clone().to(device))
            latest_model1_index = (latest_model1_index + 2 * T) % (
                2 * T * procs_per_machine
            )

        model2s = torch.stack(model2s)
        updated = torch.mean(model2s, dim=0)

        # 对于resnet18模型来说
        # Total parameters: 11689512 model.parameters()
        # Total parameters: 11699132 model.state_dict().items()

        total_params = sum(p.numel() for _, p in self.model.state_dict().items())
        logging.info(f"total_params: {total_params}")
        new_state_dict = dict()
        start_index = 0
        for i, key in enumerate(self.model.state_dict()):
            end_index = start_index + self.lens[i]
            new_state_dict[key] = updated[start_index:end_index].reshape(self.shapes[i])
            start_index = end_index

        self.model.load_state_dict(new_state_dict)

    def _averaging_by_shared_tensor_with_noesisfed(
        self, shared_tensor, current_idx, procs_per_machine, T, center_dists
    ):
        device = torch.device("cpu")
        latest_model1_index = (current_idx - 2 + 2 * T) % (2 * T)  # 第0个节点
        # 找到所有的中间模型
        model2s = []
        for _ in range(procs_per_machine):
            model2s.append(shared_tensor[latest_model1_index + 1].clone().to(device))
            latest_model1_index = (latest_model1_index + 2 * T) % (
                2 * T * procs_per_machine
            )
        model2s = torch.stack(model2s)

        # 使用KMeans算法对球心距进行聚类
        center_dists = [center_dist.cpu().numpy() for center_dist in center_dists]
        center_dists = np.array(center_dists).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(center_dists)


        from collections import Counter
        labels = kmeans.labels_
        counts = Counter(labels)
        max_class = max(counts, key=counts.get)
        indices = np.where(labels == max_class)[0]
        logging.info(f"indices: {indices}")
        updated = torch.mean(model2s[indices], dim=0)

        # 对于resnet18模型来说
        # Total parameters: 11689512 model.parameters()
        # Total parameters: 11699132 model.state_dict().items()

        total_params = sum(p.numel() for _, p in self.model.state_dict().items())
        logging.info(f"total_params: {total_params}")
        new_state_dict = dict()
        start_index = 0
        for i, key in enumerate(self.model.state_dict()):
            end_index = start_index + self.lens[i]
            new_state_dict[key] = updated[start_index:end_index].reshape(self.shapes[i])
            start_index = end_index

        self.model.load_state_dict(new_state_dict)


    def _averaging_by_shared_tensor_with_flame(
        self, shared_tensor, current_idx, procs_per_machine, T
    ):
        device = torch.device("cpu")
        latest_model1_index = (current_idx - 2 + 2 * T) % (2 * T)  # 第0个节点
        # 找到所有其他节点的初始模型以及训练后的模型
        other_model1s = []
        other_model2s = []
        for i in range(procs_per_machine):
            if i == self.rank:
                my_model1 = shared_tensor[latest_model1_index].clone().to(device)
                my_model2 = shared_tensor[latest_model1_index + 1].clone().to(device)
            else:
                other_model1s.append(
                    shared_tensor[latest_model1_index].clone().to(device)
                )
                other_model2s.append(
                    shared_tensor[latest_model1_index + 1].clone().to(device)
                )
            latest_model1_index = (latest_model1_index + 2 * T) % (
                2 * T * procs_per_machine
            )

        logging.info(f"other_model1s: {len(other_model1s)}, other_model2s: {len(other_model2s)}, my_model1: {torch.norm(my_model1)}, my_model2: {torch.norm(my_model2)}")
        # 利用Flame计算更新量
        updated = self.flame(
            other_model2s, other_model1s, procs_per_machine, my_model1, my_model2
        )

        total_params = sum(p.numel() for _, p in self.model.state_dict().items())
        logging.info(f"total_params: {total_params}")
        new_state_dict = dict()
        start_index = 0
        for i, key in enumerate(self.model.state_dict()):
            end_index = start_index + self.lens[i]
            new_state_dict[key] = updated[start_index:end_index].reshape(self.shapes[i])
            start_index = end_index

        self.model.load_state_dict(new_state_dict)

    def get_data_to_send(self, *args, **kwargs):
        self._pre_step()
        data = self.serialized_model()
        data["iteration"] = self.communication_round
        return data

    def flame(
        self, other_model2s, other_model1s, procs_per_machine, my_model1, my_model2
    ):
        # === clustering ===
        other_model2s = torch.stack(other_model2s).double()
        cluster = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=procs_per_machine // 2 + 1,
            min_samples=1,
            allow_single_cluster=True,
        )
        cluster.fit(other_model2s)
        predict_good = []
        for i, j in enumerate(cluster.labels_):
            if j == 0:
                predict_good.append(i)
        k = len(predict_good)

        # === median clipping ===
        other_model1s = torch.stack(other_model1s).double()
        model_updates = other_model2s[predict_good] - other_model1s[predict_good]
        local_norms = torch.norm(model_updates, dim=1)
        S_t = torch.median(local_norms)
        scale = S_t / local_norms
        scale = torch.where(scale > 1, torch.ones_like(scale), scale)
        model_updates = model_updates * scale.view(-1, 1)

        # === aggregating ===
        trained_params = my_model1 + model_updates
        trained_params = trained_params.sum(dim=0) / k

        # === noising ===
        delta = 1 / (procs_per_machine**2)
        epsilon = 15000
        lambda_ = 1 / epsilon * (math.sqrt(2 * math.log((1.25 / delta))))
        sigma = lambda_ * S_t.numpy()
        logging.info(f"model_updates: {model_updates.shape}")
        logging.info(
            f"sigma: {sigma}; #clean models / clean models: {k} / {predict_good}, median norm: {S_t},"
        )
        trained_params.add_(torch.normal(0, sigma, size=trained_params.size()))

        return (trained_params.float() + my_model2) / 2  # 聚合后的模型

    def _averaging_by_shared_tensor_with_foolsgold(
        self, shared_tensor, current_idx, procs_per_machine, T
    ):
        device = torch.device("cpu")
        latest_model1_index = (current_idx - 2 + 2 * T) % (2 * T)  # 第0个节点
        # 找到所有节点的初始模型
        other_model1s = []
        other_model2s = []
        for i in range(procs_per_machine):
            if i == self.rank:
                my_model1 = shared_tensor[latest_model1_index].clone().to(device)
                my_model2 = shared_tensor[latest_model1_index + 1].clone().to(device)
            else:
                other_model1s.append(
                    shared_tensor[latest_model1_index].clone().to(device)
                )
                other_model2s.append(
                    shared_tensor[latest_model1_index + 1].clone().to(device)
                )
            latest_model1_index = (latest_model1_index + 2 * T) % (
                2 * T * procs_per_machine
            )
        logging.info(f"other_model1s: {len(other_model1s)}, other_model2s: {len(other_model2s)}, my_model1: {torch.norm(my_model1)}, my_model2: {torch.norm(my_model2)}")

        model_updates = [m2 - m1 for m1, m2 in zip(other_model1s, other_model2s)]

        # 利用Foolsgold计算更新量
        updated = self.foolsgold(model_updates, my_model1)
        total_params = sum(p.numel() for _, p in self.model.state_dict().items())
        logging.info(f"total_params: {total_params}")
        new_state_dict = dict()
        start_index = 0
        for i, key in enumerate(self.model.state_dict()):
            end_index = start_index + self.lens[i]
            new_state_dict[key] = updated[start_index:end_index].reshape(self.shapes[i])
            start_index = end_index
        self.model.load_state_dict(new_state_dict)

    def foolsgold(self, model_updates, my_model1):
        model_updates = torch.stack(model_updates)
        K = len(model_updates)  # 用户数
        cs = smp.cosine_similarity(model_updates.numpy()) - np.eye(K)
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
        wv[(wv == 1)] = 0.99

        # === Logit function ===
        wv = np.log(wv / (1 - wv)) + 0.5
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        # === calculate global update ===
        logging.info(f"wv.shape: {wv.shape}")
        logging.info("wv: " + ", ".join(map(str, wv)))

        tmp = None
        for i, j in enumerate(range(len(wv))):
            if i == 0:
                tmp = model_updates[j] * wv[j]
            else:
                tmp += model_updates[j] * wv[j]

        global_update = 1 / len(wv) * tmp

        return (my_model1 + global_update + my_model1) / 2

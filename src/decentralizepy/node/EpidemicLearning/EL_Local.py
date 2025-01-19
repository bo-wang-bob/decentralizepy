import importlib
import json
import logging
import math
import os,gc
from collections import deque
from random import Random

import torch
from matplotlib import pyplot as plt
from torch.nn.utils import parameters_to_vector

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node

import tracemalloc




class EL_Local(Node):
    """
    This class defines the node on overlay graph

    """
    def save_plot(self, l, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        y_axis = [l[key] for key in l.keys()]
        x_axis = list(map(int, l.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)

    def get_neighbors(self, node=None):
        return set(self.rng.sample(self.my_neighbors, self.degree))

    def receive_DPSGD(self):
        return self.receive_channel("DPSGD", block=True)

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.my_neighbors:
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1
        self.rng = Random()
        self.rng.seed(self.dataset.random_seed + self.uid)

        logging.info("Connected to all neighbors")
        logging.info("Total number of neighbor: {}".format(len(self.my_neighbors)))


        ran = [(0, int(0.2*self.iterations)), (int(0.4*self.iterations), int(0.6*self.iterations)), (int(0.8*self.iterations), self.iterations)]
        # ran = [(0, self.iterations)]
        do_attack = False

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            self.iteration = iteration
            
            
            if self.is_malicous and iteration >= self.attack_start:
                do_attack = True

            # data = self.sharing.serialized_model()
            # state_dict = self.sharing.deserialized_model(data)
            total_params = sum(p.numel() for _, p in self.model.state_dict().items())
            model1 = torch.empty(total_params, dtype=torch.float32)
            offset = 0
            with torch.no_grad():
                for _, param in self.model.state_dict().items():
                    numel = param.numel()
                    model1[offset:offset + numel] = param.flatten()
                    offset += numel
            # assert len(self.shared_tensor[2*self.rank]) >= total_params
           
            # self.shared_tensor[2*self.rank].copy_(flat)

            # to_send0=self.sharing.get_data_to_send()
            # to_send0["CHANNEL"] = "DPSGD"
            # to_send0["iteration"] = self.iteration

            # insert_model_history(self.model_history,self.rank,self.iteration,0,to_send0)

            if not os.path.exists(f"model_{self.uid}"):
                os.mkdir(f"model_{self.uid}")
            
            
            for i in range(len(ran)):
                if iteration>=ran[i][0] and iteration<ran[i][1]:
                    torch.save(self.model.state_dict(),f"model_{self.uid}/params_{iteration}_0.pt")
                    gc.collect()
                    break
            
            self.trainer.train(self.dataset, do_attack)  # Train the model \theta_i^{t+1/2}

            for i in range(len(ran)):
                if iteration>=ran[i][0] and iteration<ran[i][1]:  
                    torch.save(self.model.state_dict(),f"model_{self.uid}/params_{iteration}_1.pt")
                    gc.collect()
                    break

            neighbors_this_round = (self.get_neighbors())  # Randomly select self.degree neighbors to communicate with

            logging.info("Neighbors this round: %s", neighbors_this_round)

            # 存储训练后的模型至共享内存
            model2 = torch.empty(total_params, dtype=torch.float32)
            offset = 0
            with torch.no_grad():
                for _, param in self.model.state_dict().items():
                    numel = param.numel()
                    model2[offset:offset + numel] = param.flatten()
                    offset += numel
            # assert len(self.shared_tensor[2*self.rank]) >= total_params

            self.history_queue.add_to_queue(model1, model2)

            logging.info("rank:{}".format(self.rank))


            logging.info("Sending has been completed!")

            self.model_history_barrier.wait() # lock -> barrier

            logging.info("Receiving has been completed!")

            if (self.iteration + 1) >= self.T:
                logging.info("Start calculating the center of the hypersphere!")
                radius = self.shared_tensor_radius[self.rank]
                dist, _ = utils.distance_calculate(model2 - model1, model1, self.shared_tensor_center[self.rank].clone())
                logging.info(f"current distance: {dist}, radius: {radius}")
                if (self.iteration - self.last_calculate_iteration) >= self.T or dist > radius:
                    # 计算自己的球心以及球半径
                    model1s = self.history_queue.get_all_model1s() # 首个为最新的模型
                    model2s = self.history_queue.get_all_model2s() 
                    grads = [m2 - m1 for m1, m2 in zip(model1s, model2s)]
                    center, radius = utils.superball_calculate(model1s, grads, self.T)
                    self.shared_tensor_center[self.rank].copy_(center)
                    self.shared_tensor_radius[self.rank].copy_(radius)


                logging.info(f"Calculating has been completed, radius: {radius}, waiting others")
                self.center_radius_barrier.wait()
                logging.info("All calculating has been completed")

                # 计算自己与其他人的球心距
                center_dists = []
                for i in sorted(self.my_neighbors):
                    center_dists.append(torch.norm(center.to(self.device)- self.shared_tensor_center[i].clone().to(self.device)))

                logging.info(f"center distance: {center_dists}")

            # 这里增加安全聚合机制

            self.sharing._averaging_by_shared_tensor(self.shared_tensor_model_history, self.history_queue.current_index, len(self.shared_tensor_model_history) //  (2 * self.T), self.T)

            

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "poisoned_test_acc": {},
                    "poisoned_test_loss": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                    "received_this_round": {},
                }

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change  # change 用于减缓测试频率
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl

                # if self.is_malicous:
                logging.info("Evaluating on poisoned test set.")
                ta, tl = self.dataset.poisoned_test(self.model, self.loss)
                results_dict["poisoned_test_acc"][iteration + 1] = ta/100
                results_dict["poisoned_test_loss"][iteration + 1] = tl
                self.save_plot(results_dict["poisoned_test_acc"],"poisoned_test_acc","Poison Accpetance","Communication Rounds"
                                ,os.path.join(self.log_dir,"{}_poison_Acceptance.png".format(self.rank)))

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)

        # self.disconnect_neighbors()
        logging.info("Storing final weight")
        torch.save(self.model.state_dict(),f"model_{self.uid}/params_{iteration}_final.pt")
        logging.info("All neighbors disconnected. Process complete!")

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        train_evaluate_after,
        reset_optimizer,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy are calculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = None

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        *args,
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )

        # reset the poisoned_train_dir & poisoned_test_dir if not malicious
        # if not self.is_malicous:
        #     config["DATASET"]["poisoned_train_dir"] = ""
        #     config["DATASET"]["poisoned_test_dir"] = ""
        # else:
        config["DATASET"]["attack_method"] = self.attack_method
        config["DATASET"]["gradmask_ratio"] = self.gradmask_ratio
        config["TRAIN_PARAMS"]["attack_method"] = self.attack_method
        config["TRAIN_PARAMS"]["gradmask_ratio"] = self.gradmask_ratio

        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        # for neighbor in self.my_neighbors:
        #     self.model_history[neighbor] = dict()

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.peer_deques0=dict()
        # self.connect_neighbors()

    def __init__(
        self,
        shared_tensor_model_history,
        model_history_barrier,
        shared_tensor_center,
        shared_tensor_radius,
        center_radius_barrier,
        T,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        is_malicous=False,
        attack_method="",
        gradmask_ratio=1.0,
        attack_start=0,
        *args,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        
        tracemalloc.start()

        self.init_log(log_dir, rank, log_level)
        
        self.shared_tensor_model_history = shared_tensor_model_history
        self.model_history_barrier = model_history_barrier
        self.shared_tensor_center = shared_tensor_center
        self.shared_tensor_radius = shared_tensor_radius
        self.center_radius_barrier = center_radius_barrier

        # logging.info(f"{type(self.shared_tensor_model_history)}, {type(shared_tensor_model_history)}")
        # logging.info(f"{type(self.model_history_barrier)}, {type(model_history_barrier)}")
        # logging.info(f"{type(self.shared_tensor_center)}, {type(shared_tensor_model_history)}")
        # logging.info(f"{type(self.shared_tensor_radius)}, {type(shared_tensor_center)}")
        # logging.info(f"{type(self.shared_tensor_radius)}, {type(shared_tensor_radius)}")
        # logging.info(f"{type(self.center_radius_barrier)}, {type(center_radius_barrier)}")


        self.is_malicous = is_malicous  # Malicious node or not
        self.attack_method = attack_method
        self.gradmask_ratio = gradmask_ratio
        self.attack_start = attack_start
        self.lr = config["OPTIMIZER_PARAMS"]["lr"]

        self.T = T # 保存的历史轮数
    
        self.history_queue = utils.my_queue(self.shared_tensor_model_history, (rank * 2 * self.T, (rank + 1) * 2 * T))
        self.last_calculate_iteration = 0

        # logging.info("Malicious: {}".format(self.is_malicous))
        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.procs_per_machine), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            *args,
        )

        nodeConfigs = config["NODE"]
        self.degree = (
            nodeConfigs["graph_degree"] if "graph_degree" in nodeConfigs else 2
        )

        logging.info(f"rank: {self.rank}, T: {self.T}, malicious: {self.is_malicous}")

        self.run()

    def __del__(self):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        logging.info("[ Top 10 ]")
        for stat in top_stats[:10]:
            logging.info(stat)
        

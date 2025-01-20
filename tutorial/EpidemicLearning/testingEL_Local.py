import logging
from pathlib import Path
from shutil import copy

from localconfig import LocalConfig
from torch import multiprocessing as mp

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Linear import Linear
from decentralizepy.node.EpidemicLearning.EL_Local import EL_Local
import torch
from torchvision import models
from torchvision.models import ResNet18_Weights


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config


if __name__ == "__main__":
    args = utils.get_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    config = read_ini(args.config_file)
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    copy(args.config_file, args.log_dir)
    copy(args.graph_file, args.log_dir)
    utils.write_args(args, args.log_dir)

    g = Graph()
    g.read_graph_from_file(args.graph_file, args.graph_type)
    n_machines = args.machines
    procs_per_machine = args.procs_per_machine[0]

    l = Linear(n_machines, procs_per_machine)
    m_id = args.machine_id
    print("###############Machine ID: ", m_id)
    malicous_nodes = list(range(args.malicious_nodes))
    attack_method = args.attack_method if len(malicous_nodes) != 0 else ""
    gradmask_ratio = args.gradmask_ratio if attack_method.lower() == "neurotoxin" else 1
    attack_start = args.attack_start if args.attack_start != 0 else args.iterations
    defense_method = args.defense_method

    print(
        f"malicous_nodes: {malicous_nodes}, attack-method: {attack_method}, gradmask_ratio: {gradmask_ratio}, defense_method: {defense_method}"
    )

    # 创建共享的tensor变量 用来存储历史模型 包括每轮聚合前和聚合后的
    T = args.history_stored
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    total_params = sum(p.numel() for _, p in model.state_dict().items())
    shared_tensor_model_history = torch.zeros(T * 2 * procs_per_machine, total_params)
    shared_tensor_model_history.share_memory_()
    model_history_barrier = mp.Barrier(
        procs_per_machine
    )  # 相应的锁变量 用来控制所有节点都把模型存入

    # 创建共享的tensor变量 用来存储每个节点计算出来的意图球心以及半径 为了节省速度，使得所有节点同时计算自身的意图球心和半径
    shared_tensor_center = torch.zeros(procs_per_machine, total_params)
    shared_tensor_radius = torch.zeros(procs_per_machine)
    shared_tensor_center.share_memory_()
    shared_tensor_radius.share_memory_()
    center_radius_barrier = mp.Barrier(
        procs_per_machine
    )  # 相应的锁变量 用来控制所有节点都计算出自身的意图球心和半径

    # print(f"{type(shared_tensor_model_history)}, {type(shared_tensor_center), {type(model_history_barrier)}}")
    processes = []
    for r in range(procs_per_machine):
        if r in malicous_nodes:
            processes.append(
                mp.Process(
                    target=EL_Local,
                    args=[
                        shared_tensor_model_history,
                        model_history_barrier,
                        shared_tensor_center,
                        shared_tensor_radius,
                        center_radius_barrier,
                        defense_method,
                        T,
                        r,
                        m_id,
                        l,
                        g,
                        my_config,
                        args.iterations,
                        args.log_dir,
                        args.weights_store_dir,
                        log_level[args.log_level],
                        args.test_after,
                        args.train_evaluate_after,
                        args.reset_optimizer,
                        True,
                        attack_method,
                        gradmask_ratio,
                        attack_start,
                    ],
                )
            )
        else:
            processes.append(
                mp.Process(
                    target=EL_Local,
                    args=[
                        shared_tensor_model_history,
                        model_history_barrier,
                        shared_tensor_center,
                        shared_tensor_radius,
                        center_radius_barrier,
                        defense_method,
                        T,
                        r,
                        m_id,
                        l,
                        g,
                        my_config,
                        args.iterations,
                        args.log_dir,
                        args.weights_store_dir,
                        log_level[args.log_level],
                        args.test_after,
                        args.train_evaluate_after,
                        args.reset_optimizer,
                        False,
                        attack_method,
                        gradmask_ratio,
                    ],
                )
            )

    for p in processes:
        p.start()

    for p in processes:
        p.join()

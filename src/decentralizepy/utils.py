import argparse
import datetime
import json
import os
import logging
import torch

def conditional_value(var, nul, default):
    """
    Set the value to default if nul.

    Parameters
    ----------
    var : any
        The value
    nul : any
        The null value. Assigns default if var == nul
    default : any
        The default value

    Returns
    -------
    type(var)
        The final value

    """
    if var != nul:
        return var
    else:
        return default


def remove_keys(d, keys_to_remove):
    """
    Removes given keys from the dict. Returns a new list.

    Parameters
    ----------
    d : dict
        The initial dictionary
    keys_to_remove : list
        List of keys to remove from dict

    Returns
    -------
    dict
        A new dictionary with the given keys removed.

    """
    return {key: d[key] for key in d if key not in keys_to_remove}


def get_args():
    """
    Utility to parse arguments.

    Returns
    -------
    args
        Command line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mid", "--machine_id", type=int, default=0)
    parser.add_argument("-ps", "--procs_per_machine", type=int, default=1, nargs="+")
    parser.add_argument("-ms", "--machines", type=int, default=1)
    parser.add_argument(
        "-ld",
        "--log_dir",
        type=str,
        default="./{}".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    parser.add_argument(
        "-wsd",
        "--weights_store_dir",
        type=str,
        default="./{}_ws".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    parser.add_argument("-is", "--iterations", type=int, default=1)
    parser.add_argument("-cf", "--config_file", type=str, default="config.ini")
    parser.add_argument("-ll", "--log_level", type=str, default="INFO")
    parser.add_argument("-gf", "--graph_file", type=str, default="36_nodes.edges")
    parser.add_argument("-gt", "--graph_type", type=str, default="edges")
    parser.add_argument("-ta", "--test_after", type=int, default=5)
    parser.add_argument("-tea", "--train_evaluate_after", type=int, default=1)
    parser.add_argument("-ro", "--reset_optimizer", type=int, default=1)
    parser.add_argument("-sm", "--server_machine", type=int, default=0)
    parser.add_argument("-sr", "--server_rank", type=int, default=-1)
    parser.add_argument("-wr", "--working_rate", type=float, default=1.0)
    parser.add_argument("-mals", "--malicious_nodes", type=int, default=0)
    parser.add_argument("-am", "--attack_method", type=str, default="")
    parser.add_argument("-gr", "--gradmask_ratio", type=float, default=1.0)
    parser.add_argument("-as", "--attack_start", type=int, default=0)
    parser.add_argument("-T", "--history_stored", type=int, default=5)

    args = parser.parse_args()
    return args


def write_args(args, path):
    """
    Write arguments to a json file

    Parameters
    ----------
    args : args
        Command line args
    path : str
        Location of the file to write to

    """
    data = {
        "machine_id": args.machine_id,
        "procs_per_machine": args.procs_per_machine,
        "machines": args.machines,
        "log_dir": args.log_dir,
        "weights_store_dir": args.weights_store_dir,
        "iterations": args.iterations,
        "config_file": args.config_file,
        "log_level": args.log_level,
        "graph_file": args.graph_file,
        "graph_type": args.graph_type,
        "test_after": args.test_after,
        "train_evaluate_after": args.train_evaluate_after,
        "reset_optimizer": args.reset_optimizer,
        "working_rate": args.working_rate,
    }
    with open(os.path.join(path, "args.json"), "w") as of:
        json.dump(data, of)


def identity(obj):
    """
    Identity function
    Parameters
    ----------
    obj
        Some object
    Returns
    -------
     obj
        The same object
    """
    return obj



class my_queue:
    def __init__(self, shared_tensor, ran):
        self.current_index = 0
        self.shared_tensor = shared_tensor
        self.left_ran = ran[0] # 共享内存上属于自己的部分，左开右闭
        self.right_ran = ran[1]
        self.queue_size = ran[1] - ran[0]
        self.current_size = 0
        logging.info(f"left_ran: {self.left_ran} right_ran: {self.right_ran}")


    def add_to_queue(self, model1, model2):
        self.shared_tensor[self.left_ran + self.current_index].copy_(model1)
        self.shared_tensor[self.left_ran + self.current_index + 1].copy_(model2)
        self.current_index = (self.current_index + 2) % self.queue_size
        if self.current_size < self.queue_size:
            self.current_size += 2
    
        logging.info(f"current index: {self.current_index}, current size: {self.current_size}")


    def get_latest_model1(self):
        lastest_index = (self.current_index - 2 + self.queue_size) % self.queue_size
        return self.shared_tensor[self.left_ran + lastest_index].clone()
    
    
    def get_latest_model2(self):
        lastest_index = (self.current_index - 2 + self.queue_size) % self.queue_size
        return self.shared_tensor[self.left_ran + lastest_index + 1].clone()


    def get_all_model1s(self):
        all_model1s = []
        size = self.current_size
        lastest_index = (self.current_index - 2 + self.queue_size) % self.queue_size
        while size != 0 :
            all_model1s.append(self.shared_tensor[self.left_ran + lastest_index].clone())
            size -= 2
            lastest_index = (lastest_index - 2 + self.queue_size) % self.queue_size   
        return all_model1s
    
    def get_all_model2s(self):
        all_model2s = []
        size = self.current_size
        lastest_index = (self.current_index - 2 + self.queue_size) % self.queue_size
        while size != 0 :
            all_model2s.append(self.shared_tensor[self.left_ran + lastest_index + 1].clone())
            size -= 2
            lastest_index = (lastest_index - 2 + self.queue_size) % self.queue_size
        return all_model2s
        


def distance_calculate(k, b, center, epsilon=1e-8):
    """
    计算点到直线的距离，并返回距离和投影点。

    参数:
    k (torch.Tensor): 直线的方向向量。
    b (torch.Tensor): 直线上的一个点。
    center (torch.Tensor): 要计算距离的点。
    epsilon (float): 用于数值稳定性的小值。

    返回:
    dis (torch.Tensor): 点到直线的距离。
    pt (torch.Tensor): 点在直线上的投影点。
    """ 
    w = center-b
    alpha = torch.dot(k,w) / (torch.dot(k,k) + epsilon)
    if alpha >= 0:
        pt = b + alpha * k
        dis = torch.norm(pt - center)
    else:
        dis = torch.norm(w)
        pt = b
    return dis, pt



def superball_calculate(model_history, grad_history, T):
    """
    模拟退火求覆盖射线集的超球
    
    """
    T = 5
    tao = 100 #10000
    TAO_0 = 1e-6
    ALPHA = 0.98
    ZETA = 0.8
    model_history_tensor = torch.stack(model_history)
    center = torch.mean(model_history_tensor, dim=0)
    logging.info(f"center: {center}")
    dis_list = list()
    for i in range(T):
        k=grad_history[i]
        b=model_history[i]
        dis, pt=distance_calculate(k,b,center)
        dis_list.append((dis, pt))

    dis_list.sort(key=lambda x:x[0].item())
    radius = dis_list[int(T*ZETA)][0]

    cnt = 0
    while tao > TAO_0:
        # logging.info(f"cnt: {cnt}, dis_list: {dis_list}")
        mpt=dis_list[-1][1]
        acenter = center + tao*((mpt-center)/torch.norm(mpt-center))
        dis_list = list()
        for i in range(T):
            k=grad_history[i]
            b=model_history[i]
            dis,pt=distance_calculate(k,b,acenter)
            dis_list.append((dis,pt))

        dis_list.sort(key=lambda x:x[0].item())   
        aradius = dis_list[int(T*ZETA)][0]
        if aradius < radius:
            center = acenter
            radius = aradius
            cnt += 1    
        else:
            p = torch.exp((radius-aradius)/tao)
            if torch.rand(1).item() < p.item():
                center = acenter
                radius = aradius
                cnt += 1
        tao *= ALPHA

    logging.info(f"T: {T}, Simulated Annealing Cnt: {cnt}, Radius: {radius}")

    return center,radius
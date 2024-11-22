import logging

import torch
import copy
import numpy as np


from decentralizepy import utils

import itertools


class Training:
    """
    This class implements the training module for a single node.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss,
        log_dir,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
        attack_method="",
        gradmask_ratio=1,
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
        model : torch.nn.Module
            Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        log_dir : str
            Directory to log the model change.
        rounds : int, optional
            Number of steps/epochs per training call
        full_epochs : bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.

        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.log_dir = log_dir
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.rounds = utils.conditional_value(rounds, "", int(1))
        self.full_epochs = utils.conditional_value(full_epochs, "", False)
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)
        self.attack_method = attack_method
        self.gradmask_ratio = gradmask_ratio

    def reset_optimizer(self, optimizer):
        """
        Replace the current optimizer with a new one

        Parameters
        ----------
        optimizer : torch.optim
            A new optimizer

        """
        self.optimizer = optimizer

    def eval_loss(self, dataset):
        """
        Evaluate the loss on the training set

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                count += 1
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        return loss

    def trainstep(self, data, target):
        """
        One training step on a minibatch.

        Parameters
        ----------
        data : any
            Data item
        target : any
            Label

        Returns
        -------
        int
            Loss Value for the step

        """
        self.model.zero_grad()
        output = self.model(data)
        loss_val = self.loss(output, target)
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()

    def train_full(self, dataset):
        """
        One training iteration, goes through the entire dataset

        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.

        """
        for epoch in range(self.rounds):
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                logging.debug(
                    "Starting minibatch {} with num_samples: {}".format(
                        count, len(data)
                    )
                )
                logging.debug("Classes: {}".format(target))
                epoch_loss += self.trainstep(data, target)
                count += 1
            logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))

    def train(self, dataset, do_attack=False):
        """
        One training iteration

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        self.model.train()
        logging.info("do_attack: {}".format(do_attack))
        if not do_attack:
            if self.full_epochs:
                self.train_full(dataset)
            else:
                iter_loss = 0.0
                count = 0
                trainset = dataset.get_trainset(self.batch_size, self.shuffle)
                while count < self.rounds:
                    for data, target in trainset:
                        iter_loss += self.trainstep(data, target)
                        count += 1
                        logging.debug(
                            "Round: {} loss: {}".format(count, iter_loss / count)
                        )
                        if count >= self.rounds:
                            break
        else:
            logging.info("Starting attack")
            mask_grad_list = None
            if self.attack_method.lower() == "neurotoxin":
                assert self.gradmask_ratio != 1
                cleanset_under_neurotoxin = dataset.get_cleanset_under_neurotoxin(
                    self.batch_size, self.shuffle
                )
                model_under_neurotoxin = copy.deepcopy(self.model)
                model_under_neurotoxin.train()
                model_under_neurotoxin.zero_grad()
                for data, target in cleanset_under_neurotoxin:
                    output = model_under_neurotoxin(data)
                    loss_val = self.loss(output, target)
                    loss_val.backward()

                mask_grad_list = []
                grad_list = []
                grad_abs_sum_list = []
                k_layer = 0
                for _, parms in model_under_neurotoxin.named_parameters():
                    if parms.requires_grad:
                        grad_list.append(parms.grad.abs().view(-1))
                        grad_abs_sum_list.append(
                            parms.grad.abs().view(-1).sum().item()
                        )  # sum of absolute values of gradients
                        k_layer += 1
                grad_list = torch.cat(grad_list)  # concatenate all gradients
                _, indices = torch.topk(
                    -1 * grad_list, int(len(grad_list) * self.gradmask_ratio)
                )  # get indices of top k gradients 这里的indices实际上是绝对值最小的k个梯度的索引
                mask_flat_all_layer = torch.zeros(len(grad_list))
                mask_flat_all_layer[indices] = 1.0

                count = 0
                percentage_mask_list = []
                k_layer = 0
                grad_abs_percentage_list = []
                for _, parms in model_under_neurotoxin.named_parameters():
                    if parms.requires_grad:
                        gradients_length = len(parms.grad.abs().view(-1))
                        mask_flat = mask_flat_all_layer[
                            count : count + gradients_length
                        ]
                        mask_grad_list.append(mask_flat.reshape(parms.grad.size()))
                        count += gradients_length
                        percentage_mask1 = (
                            mask_flat.sum().item() / float(gradients_length) * 100.0
                        )
                        percentage_mask_list.append(percentage_mask1)
                        grad_abs_percentage_list.append(
                            grad_abs_sum_list[k_layer] / np.sum(grad_abs_sum_list)
                        )
                        k_layer += 1
                logging.info(
                    "Percentage of gradients masked: {}".format(percentage_mask_list)
                )
                logging.info(
                    "Percentage of gradients masked: {}".format(
                        grad_abs_percentage_list
                    )
                )

            iter_loss = 0.0
            count = 0
            trainset = dataset.get_trainset_under_attack(self.batch_size, self.shuffle)
            poisonset = dataset.get_poisoned_trainset(self.batch_size, self.shuffle)

            trainset_iter = iter(trainset)
            poisonset_iter = itertools.cycle(poisonset)
            # logging.info("Poisoned set: {}".format(len(poisonset)))
            while count < self.rounds:
                for data, target in trainset_iter:
                    poison_data, poison_target = next(poisonset_iter)

                    data = torch.cat((data, poison_data))
                    target = torch.cat((target, poison_target))
                    self.model.zero_grad()
                    output = self.model(data)
                    loss_val = self.loss(output, target)
                    loss_val.backward()
                    iter_loss += loss_val.item()
                    count += 1
                    logging.debug("Round: {} loss: {}".format(count, iter_loss / count))
                    if count >= self.rounds:
                        break

            if self.attack_method.lower() == "neurotoxin":
                mask_grad_list_iter = iter(mask_grad_list)
                for _, parms in self.model.named_parameters():
                    if parms.requires_grad:
                        mask_grad = next(mask_grad_list_iter)
                        parms.grad = parms.grad * mask_grad

            self.optimizer.step()

        logging.info("Training done")

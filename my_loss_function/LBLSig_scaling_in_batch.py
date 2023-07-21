from my_customs import *

from abc import ABC, abstractmethod

import json
import torch
import numpy
import time
import copy
import scipy.io
import os

import time


class LBLSig_scaling_in_batch(ABC):

    def __init__(self, loss_function: str, net_name: str):

        super().__init__()

        assert loss_function in ('LBLSig_scaling_in_batch', ), 'Loss function must be "LBLSig_in_batch".'
        self.loss_function = loss_function

        self.R_square = None  # the square of the hypersphere radius R
        self.c = None  # hypersphere center c

        self.net_name = net_name
        self.net = None  # neural network \phi

        self.optimizer_parameters = None

        # self.lambda_error = None

        self.scaling_parameter = None

        self.mu = None
        # 这个参数跟WSI-AE里面的一样，是最后选择决策阈值时的那个抛弃多少样本的百分比，
        # 这里我们就统一nu_for_R_optimization和mu相等。
        self.theta = None  # 决策阈值

        self.training_accuracy = None

        self.warm_up_n_epochs = None

    @abstractmethod
    def build_network(self, *args, **kwargs):
        # 因为有多种不同类型的网络
        raise SystemExit('Not Implemented.')

    @abstractmethod
    def compute_hypersphere_center(self, *args, **kwargs):
        # 不同类型的网络计算球心的方式不同
        raise SystemExit('Not Implemented.')

    def train(self, train_dataset, optimizer_parameters: My_Optimizer_Parameters, scaling_parameter, mu,
              warm_up_n_epochs):

        self.R_square = torch.tensor(0.0, device=optimizer_parameters.device)

        self.scaling_parameter = scaling_parameter

        self.mu = mu

        self.warm_up_n_epochs = warm_up_n_epochs

        self.optimizer_parameters = optimizer_parameters  # 包含优化器的一些参数

        self.net = self.net.to(optimizer_parameters.device)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=optimizer_parameters.batch_size,
                                                       shuffle=True,
                                                       num_workers=optimizer_parameters.number_of_workers)
        if optimizer_parameters.optimizer_name.lower() == 'adam':
            train_optimizer = torch.optim.Adam(self.net.parameters(),
                                               lr=optimizer_parameters.learning_rate,
                                               weight_decay=optimizer_parameters.weight_decay)
        else:
            raise SystemExit('Unknown optimizer')

        if optimizer_parameters.learning_rate_milestones != []:
            train_scheduler = torch.optim.lr_scheduler.MultiStepLR(train_optimizer,
                                                                   milestones=optimizer_parameters.learning_rate_milestones,
                                                                   gamma=0.1)
        else:
            # 表示不需要进行milestones衰减
            train_scheduler = torch.optim.lr_scheduler.MultiStepLR(train_optimizer,
                                                                   milestones=[0],
                                                                   gamma=1)

        # region Initialize hypersphere center c

        self.compute_hypersphere_center(train_dataloader)

        # endregion

        self.net.train()

        for k_epoch in range(optimizer_parameters.number_of_epochs):

            for k_batch_index, k_train_dataloader in enumerate(train_dataloader):

                k_batch_inputs, _ = k_train_dataloader  # 这里两个变量虽然是同一个地址，但是下面进入gpu的时候制作了副本
                k_batch_inputs = k_batch_inputs.to(optimizer_parameters.device)

                train_optimizer.zero_grad()

                k_batch_outputs = self.net(k_batch_inputs)

                distance_errors = torch.sum((k_batch_outputs - self.c) ** 2, dim=tuple(range(1, k_batch_outputs.dim())))  # 注意，网络输出不仅仅只是向量，也有可能是张量

                if k_epoch >= self.warm_up_n_epochs:

                    self.R_square = torch.quantile(distance_errors, 1 - self.mu)  # 这个函数是在pytorch 1.7才开始有的
                    self.R_square = self.R_square.detach()  # ！！！！！非常重要！！！！！

                    temp = self.R_square - distance_errors

                    ######################scaling manually
                    temp = torch.max(-10 * torch.ones_like(temp), temp)
                    loss = -torch.mean(torch.log(torch.sigmoid( scaling_parameter * temp )))
                    ######################scaling manually

                else:
                    loss = torch.mean(distance_errors)

                loss.backward()
                train_optimizer.step()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.06f}'.format(
                    k_epoch, (k_batch_index+1) * len(k_train_dataloader[0]), train_dataset.number_of_data,
                    100. * (k_batch_index+1) / len(train_dataloader), loss.item())
                )

            train_scheduler.step()

        # region 计算阈值
        self.net.eval()
        distance_errors_all = numpy.array([])
        with torch.no_grad():
            for k_train_dataloader in train_dataloader:
                k_batch_inputs, _ = k_train_dataloader
                k_batch_inputs = k_batch_inputs.to(self.optimizer_parameters.device)
                k_batch_outputs = self.net(k_batch_inputs)

                distance_errors = torch.sum((k_batch_outputs - self.c) ** 2, dim=tuple(range(1, k_batch_outputs.dim())))  # 注意，网络输出不仅仅只是向量，也有可能是张量

                distance_errors_all = numpy.append(distance_errors_all, distance_errors.cpu().numpy())

        self.theta = numpy.quantile(distance_errors_all, 1 - self.mu)

        MissClassificationRate_Training = numpy.float64(0)
        for i in range(train_dataset.number_of_data):
            if distance_errors_all[i] >= self.theta:
                MissClassificationRate_Training = MissClassificationRate_Training + 1

        self.training_accuracy = 1 - MissClassificationRate_Training / train_dataset.number_of_data

        # endregion

    def test(self, test_dataset):

        # 这里不在移动到"device"中是因为在train中已经操作过了
        # self.net = self.net.to(self.optimizer_parameters.device)

        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.optimizer_parameters.batch_size,
                                                      shuffle=False, num_workers=self.optimizer_parameters.number_of_workers)

        self.net.eval()

        distance_errors_all = numpy.array([])
        batch_targets_all = numpy.array([])
        with torch.no_grad():
            for k_test_dataloader in test_dataloader:
                k_batch_inputs, k_batch_targets = k_test_dataloader
                # k_batch_targets在Dataloader中，经过不断选取，最后返回的是一个(N,)的一维张量Tensor型。
                k_batch_inputs = k_batch_inputs.to(self.optimizer_parameters.device)
                k_batch_outputs = self.net(k_batch_inputs)

                distance_errors = torch.sum((k_batch_outputs - self.c) ** 2, dim=tuple(range(1, k_batch_outputs.dim())))  # 注意，网络输出不仅仅只是向量，也有可能是张量

                distance_errors_all = numpy.append(distance_errors_all, distance_errors.cpu().numpy())
                batch_targets_all = numpy.append(batch_targets_all, k_batch_targets.cpu().numpy())

        testing_accuracy = my_Struct()

        # region testing_accuracy.auc
        testing_accuracy.auc = my_roc_curve(distance_errors_all, batch_targets_all, test_dataset.target_label, 2)

        # endregion

        # region Conduct classification
        TV_T = batch_targets_all.flatten()
        # 在构建数据的时候，我都是强制采用numpy
        # 用了.flatten()相当于是deep copy
        # 把TY和TY_T都转换成一维数组，方便后面代码表示，只用考虑第几个元素，不用关心行还是列向量
        TY = numpy.zeros_like(TV_T)
        distance_errors_all = distance_errors_all.flatten()
        for k in range(test_dataset.number_of_data):
            if distance_errors_all[k] >= self.theta:
                TY[k] = -1
            else:
                TY[k] = test_dataset.target_label  # 虽然label是一个numpy数组格式，但是它里面只有一个元素，所以可以直接赋值，如果里面有多个元素就不行了

        # endregion

        # region testing_accuracy.overall_accuracy
        MissClassificationRate_Testing = numpy.float64(0)
        for i in range(test_dataset.number_of_data):
            if TV_T[i] != TY[i]:
                MissClassificationRate_Testing = MissClassificationRate_Testing + 1

        testing_accuracy.overall_accuracy = 1 - MissClassificationRate_Testing / test_dataset.number_of_data

        # endregion

        # region testing_accuracy.Gmean
        testing_accuracy.Gmean = numpy.float64(1)
        label_including_netative = numpy.append(test_dataset.target_label, -1)
        number_of_each_class_of_test_data = numpy.array([test_dataset.number_of_data_target_class, test_dataset.number_of_data_outlier_class])
        for k in range(label_including_netative.size):
            position = numpy.argwhere(
                TV_T == label_including_netative[k])  # 得到目标类和非目标类的位置。因为上面已经把TV_T化成了一维数组，所以二维数组position的第二维是1，即只有一列
            if position.size != 0:
                testing_accuracy.Gmean = testing_accuracy.Gmean * (
                            numpy.sum(TY[position] == label_including_netative[k]) / number_of_each_class_of_test_data[
                        k])

        testing_accuracy.Gmean = testing_accuracy.Gmean ** (1 / label_including_netative.size)

        # endregion

        # region Recall，testing_accuracy.R
        position_positive = numpy.argwhere(TV_T == test_dataset.target_label)  # 得到目标类和非目标类的位置。因为上面已经把TV_T化成了一维数组，所以二维数组position的第二维是1，即只有一列
        if position_positive.size != 0:
            testing_accuracy.R = numpy.sum(TY[position_positive] == test_dataset.target_label) / test_dataset.number_of_data_target_class

        # endregion

        # region Precision，testing_accuracy.P
        position_positive = numpy.argwhere(TV_T == test_dataset.target_label)  # 测试集等于标签的位置
        position_negative = numpy.argwhere(TV_T != test_dataset.target_label)  # position_negative
        if position_positive.size != 0 and position_negative.size != 0:
            testing_accuracy.P = numpy.sum(TY[position_positive] == test_dataset.target_label) / (
                        numpy.sum(TY[position_positive] == test_dataset.target_label) + numpy.sum(TY[position_negative] == test_dataset.target_label))

        # endregion

        # region F1,testing_accuracy.F1
        testing_accuracy.F1 = (2 * testing_accuracy.P * testing_accuracy.R) / (
                    testing_accuracy.P + testing_accuracy.R)

        # endregion

        return testing_accuracy

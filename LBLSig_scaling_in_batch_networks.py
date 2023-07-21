from my_loss_function.LBLSig_scaling_in_batch import *
from my_networks.DSLeNet import *
from my_networks.my_MLP import *
from my_networks.OCITN import *
from my_networks import PANDA_ResNet

from my_customs import *

import json
import torch
import numpy
import time
import copy
import scipy.io
import os
from PIL import Image
import torchvision

import time


class LBLSig_scaling_in_batch_DSLeNet(LBLSig_scaling_in_batch):

    def __init__(self, loss_function: str, net_name: str):

        super().__init__(loss_function, net_name)


        self.AE_net = None  # autoencoder network for pretraining

        self.AE_optimizer_parameters = None


    def build_network(self):

        # region build network
        if self.net_name in ('DSLeNet_CIFAR10', 'DSLeNet_SVHN_32'): # DeepSVDD论文中提供的网络
            self.net = DSLeNet_CIFAR10()
            self.AE_net = DSLeNet_CIFAR10_Autoencoder()

        else:
            raise SystemExit('Unknown Networks.')

        # endregion

    def compute_hypersphere_center(self, train_dataloader):

        self.c = torch.zeros(self.net.output_dimension, device=self.optimizer_parameters.device)
        number_of_data = 0
        self.net.eval()
        with torch.no_grad():
            for k_train_dataloader in train_dataloader:
                k_batch_inputs, _ = k_train_dataloader
                k_batch_inputs = k_batch_inputs.to(self.optimizer_parameters.device)
                k_batch_outputs = self.net(k_batch_inputs)
                self.c += torch.sum(k_batch_outputs, dim=0)
                number_of_data += k_batch_outputs.shape[0]

        self.c /= number_of_data

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        self.c[(abs(self.c) < 0.1) & (self.c < 0.1)] = -0.1
        self.c[(abs(self.c) < 0.1) & (self.c > 0.1)] = 0.1

    def pretrain(self, train_dataset, AE_optimizer_parameters: My_Optimizer_Parameters):
        """
        Pretrains the weights for the Deep SVDD network \phi via autoencoder.
        AE_optimizer_parameters包含的变量包括：
        1. optimizer_name
        2. learning_rate
        3. learning_rate_milestones
        4. number_of_epochs
        5. batch_size
        6. weight_decay, 即正则项的权衡系数lambda
        7. device
        8. number_of_workers
        9.其他优化器需要的参数
        """

        self.AE_optimizer_parameters = AE_optimizer_parameters

        self.AE_net = self.AE_net.to(AE_optimizer_parameters.device)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=AE_optimizer_parameters.batch_size,
                                                       shuffle=True,
                                                       num_workers=AE_optimizer_parameters.number_of_workers)
        if AE_optimizer_parameters.optimizer_name.lower() == 'adam':
            train_optimizer = torch.optim.Adam(self.AE_net.parameters(),
                                               lr=AE_optimizer_parameters.learning_rate,
                                               weight_decay=AE_optimizer_parameters.weight_decay)
        else:
            raise SystemExit('Unknown optimizer')

        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(train_optimizer,
                                                               milestones=AE_optimizer_parameters.learning_rate_milestones,
                                                               gamma=0.1)

        self.AE_net.train()

        for k_epoch in range(AE_optimizer_parameters.number_of_epochs):

            for k_batch_index, k_train_dataloader in enumerate(train_dataloader):

                k_batch_inputs, _ = k_train_dataloader
                k_batch_inputs = k_batch_inputs.to(AE_optimizer_parameters.device)

                train_optimizer.zero_grad()

                k_batch_outputs = self.AE_net(k_batch_inputs)

                temp = torch.sum((k_batch_outputs - k_batch_inputs) ** 2, dim=tuple(range(1, k_batch_outputs.dim())))
                loss = torch.mean(temp)

                loss.backward()
                train_optimizer.step()

                print('AE: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.06f}'.format(
                    k_epoch, (k_batch_index+1) * len(k_train_dataloader[0]), train_dataset.number_of_data,
                    100. * (k_batch_index+1) / len(train_dataloader), loss.item())
                )


            train_scheduler.step()

        net_dict = self.net.state_dict()
        AE_net_dict = self.AE_net.state_dict()

        # Filter out decoder network keys
        AE_net_dict = {k: v for k, v in AE_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(AE_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)
        # 注意这个函数的说明，"Copies parameters and buffers from state_dict into this module and its descendants."
        # 是'Copies'，所以即使net_dict是在cuda里，但是net还是CPU，最终输出还是CPU

    def run_with_optimal_parameters(self, save_name_results, train_dataset, test_dataset, number_of_repeats,
                                    flag_GPU: str, number_of_workers, flag_random_seed = True):

        optimal_testing_accuracy, optimal_ave_testing_accuracy, \
        optimal_AE_optimizer_parameters, optimal_optimizer_parameters, \
        optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs = \
            self.my_loadmat_for_optimal_results(
                os.path.join(save_name_results, 'optimal_results.mat'))

        optimal_AE_optimizer_parameters.device = flag_GPU
        optimal_AE_optimizer_parameters.number_of_workers = number_of_workers

        optimal_optimizer_parameters.device = flag_GPU
        optimal_optimizer_parameters.number_of_workers = number_of_workers

        testing_accuracy = numpy.array([None] * number_of_repeats)
        ave_testing_accuracy = my_Struct()
        if flag_random_seed:
            set_random_seed(1)
        for k in range(number_of_repeats):
            self.build_network()
            self.pretrain(train_dataset, optimal_AE_optimizer_parameters)
            self.train(train_dataset, optimal_optimizer_parameters, optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs)
            testing_accuracy[k] = self.test(test_dataset)

        ave_testing_accuracy.overall_accuracy = numpy.float64(0)
        ave_testing_accuracy.Gmean = numpy.float64(0)
        ave_testing_accuracy.R = numpy.float64(0)
        ave_testing_accuracy.P = numpy.float64(0)
        ave_testing_accuracy.F1 = numpy.float64(0)
        ave_testing_accuracy.auc = numpy.float64(0)
        for k in range(number_of_repeats):
            ave_testing_accuracy.overall_accuracy = ave_testing_accuracy.overall_accuracy + testing_accuracy[
                k].overall_accuracy / number_of_repeats
            ave_testing_accuracy.Gmean = ave_testing_accuracy.Gmean + testing_accuracy[k].Gmean / number_of_repeats
            ave_testing_accuracy.R = ave_testing_accuracy.R + testing_accuracy[k].R / number_of_repeats
            ave_testing_accuracy.P = ave_testing_accuracy.P + testing_accuracy[k].P / number_of_repeats
            ave_testing_accuracy.F1 = ave_testing_accuracy.F1 + testing_accuracy[k].F1 / number_of_repeats
            ave_testing_accuracy.auc = ave_testing_accuracy.auc + testing_accuracy[k].auc / number_of_repeats

        return testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy

    def my_loadmat_for_optimal_results(self, mat_path):
        my_results = scipy.io.loadmat(mat_path, mat_dtype=True, struct_as_record=False, squeeze_me=True)
        #  #struct_as_record=False, squeeze_me=True #simplify_cells=True #matlab_compatible=False
        # 这是我测试出来我觉着最好的结果

        """
        读进来的my_results包含"optimal_AE_optimizer_parameters","optimal_optimizer_parameters","optimal_mu",
        "optimal_lambda_error" (only for soft-boundary)
        "optimal_testing_accuracy","optimal_ave_testing_accuracy"

        optimal_AE_optimizer_parameters和optimal_optimizer_parameters是  My_Optimizer_Parameters类实例，包含以下变量：
        optimizer_name，
        learning_rate，
        learning_rate_milestones，
        number_of_epochs，
        batch_size，
        weight_decay，
        device，
        number_of_workers

        # optimal_AE_parameters是一个ndarray(number_of_AEs,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        # flag_GPU，int型
        # activation_function，str型
        # parameter_of_activation_function，numpy.float64型
        # number_of_hidden_layer_neurons，numpy.float64型
        # C，numpy.float64型

        optimal_testing_accuracy是一个ndarray(number_of_repeats,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        F1，numpy.float64型
        Gmean，numpy.float64型
        P，numpy.float64型
        R，numpy.float64型
        auc，numpy.float64型
        overall_accuracy，numpy.float64型

        optimal_ave_testing_accuracy是一个my_Struct结构体，包含以下变量：
        overall_accuracy，numpy.float64型
        Gmean，numpy.float64型
        R，numpy.float64型
        P，numpy.float64型
        F1，numpy.float64型
        auc，numpy.float64型
        """

        # region optimal_AE_optimizer_parameters和optimal_optimal_optimizer_parameters

        optimal_AE_optimizer_parameters = My_Optimizer_Parameters(
            my_results['optimal_AE_optimizer_parameters'].optimizer_name,
            my_results['optimal_AE_optimizer_parameters'].learning_rate,
            my_results['optimal_AE_optimizer_parameters'].learning_rate_milestones,
            my_results['optimal_AE_optimizer_parameters'].number_of_epochs,
            my_results['optimal_AE_optimizer_parameters'].batch_size,
            my_results['optimal_AE_optimizer_parameters'].weight_decay,
            my_results['optimal_AE_optimizer_parameters'].device,
            my_results['optimal_AE_optimizer_parameters'].number_of_workers,
        )

        optimal_optimizer_parameters = My_Optimizer_Parameters(
            my_results['optimal_optimizer_parameters'].optimizer_name,
            my_results['optimal_optimizer_parameters'].learning_rate,
            my_results['optimal_optimizer_parameters'].learning_rate_milestones,
            my_results['optimal_optimizer_parameters'].number_of_epochs,
            my_results['optimal_optimizer_parameters'].batch_size,
            my_results['optimal_optimizer_parameters'].weight_decay,
            my_results['optimal_optimizer_parameters'].device,
            my_results['optimal_optimizer_parameters'].number_of_workers,
        )

        # endregion

        # region optimal_testing_accuracy
        optimal_testing_accuracy = numpy.array([])
        for k_testing_accuracy in my_results['optimal_testing_accuracy']:
            optimal_testing_accuracy = numpy.append(optimal_testing_accuracy, my_Struct())

            optimal_testing_accuracy[-1].overall_accuracy = numpy.float64(k_testing_accuracy.overall_accuracy)
            optimal_testing_accuracy[-1].Gmean = numpy.float64(k_testing_accuracy.Gmean)
            optimal_testing_accuracy[-1].R = numpy.float64(k_testing_accuracy.R)
            optimal_testing_accuracy[-1].P = numpy.float64(k_testing_accuracy.P)
            optimal_testing_accuracy[-1].F1 = numpy.float64(k_testing_accuracy.F1)
            optimal_testing_accuracy[-1].auc = numpy.float64(k_testing_accuracy.auc)
        # endregion

        # region ave_testing_accuracy
        optimal_ave_testing_accuracy = my_Struct()
        optimal_ave_testing_accuracy.overall_accuracy = numpy.float64(
            my_results['optimal_ave_testing_accuracy'].overall_accuracy)
        optimal_ave_testing_accuracy.Gmean = numpy.float64(my_results['optimal_ave_testing_accuracy'].Gmean)
        optimal_ave_testing_accuracy.R = numpy.float64(my_results['optimal_ave_testing_accuracy'].R)
        optimal_ave_testing_accuracy.P = numpy.float64(my_results['optimal_ave_testing_accuracy'].P)
        optimal_ave_testing_accuracy.F1 = numpy.float64(my_results['optimal_ave_testing_accuracy'].F1)
        optimal_ave_testing_accuracy.auc = numpy.float64(my_results['optimal_ave_testing_accuracy'].auc)
        # endregion

        # region mu, lambda_error (only for soft-boundary)
        optimal_scaling_parameter = my_results['optimal_scaling_parameter']
        optimal_mu = my_results['optimal_mu']
        optimal_warm_up_n_epochs = my_results['optimal_warm_up_n_epochs']

        return optimal_testing_accuracy, optimal_ave_testing_accuracy, \
               optimal_AE_optimizer_parameters, optimal_optimizer_parameters, optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs


        # endregion


class LBLSig_scaling_in_batch_MLP(LBLSig_scaling_in_batch):

    def __init__(self, loss_function: str, net_name: str):

        super().__init__(loss_function, net_name)

    def build_network(self, number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons):

        # region build network
        if self.net_name == 'MLP_1_HL':
            self.net = MLP_1_HL(number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons)
        elif self.net_name == 'MLP_2_HLs':
            self.net = MLP_2_HLs(number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons)
            # 两个隐藏层的神经元数量相等
        else:
            raise SystemExit('Unknown Networks.')

        # endregion

    def compute_hypersphere_center(self, train_dataloader):

        self.c = torch.zeros(self.net.number_of_input_neurons, device=self.optimizer_parameters.device)
        number_of_data = 0
        self.net.eval()
        with torch.no_grad():
            for k_train_dataloader in train_dataloader:
                k_batch_inputs, _ = k_train_dataloader
                k_batch_inputs = k_batch_inputs.to(self.optimizer_parameters.device)
                self.c += torch.sum(k_batch_inputs, dim=0)
                number_of_data += k_batch_inputs.shape[0]

        self.c /= number_of_data

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        self.c[(abs(self.c) < 0.1) & (self.c < 0.1)] = -0.1
        self.c[(abs(self.c) < 0.1) & (self.c > 0.1)] = 0.1

    def run_with_optimal_parameters(self, save_name_results, train_dataset, test_dataset, number_of_repeats,
                                    flag_GPU: str, number_of_workers, flag_random_seed = True):

        optimal_testing_accuracy, optimal_ave_testing_accuracy, \
        optimal_optimizer_parameters, optimal_number_of_hidden_neurons, \
        optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs = \
            self.my_loadmat_for_optimal_results(
                os.path.join(save_name_results, 'optimal_results.mat'))

        optimal_optimizer_parameters.device = flag_GPU
        optimal_optimizer_parameters.number_of_workers = number_of_workers

        testing_accuracy = numpy.array([None] * number_of_repeats)
        ave_testing_accuracy = my_Struct()
        if flag_random_seed:
            set_random_seed(1)
        for k in range(number_of_repeats):
            self.build_network(train_dataset.data.shape[0], optimal_number_of_hidden_neurons, train_dataset.data.shape[0])
            self.train(train_dataset, optimal_optimizer_parameters, optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs)
            testing_accuracy[k] = self.test(test_dataset)

        ave_testing_accuracy.overall_accuracy = numpy.float64(0)
        ave_testing_accuracy.Gmean = numpy.float64(0)
        ave_testing_accuracy.R = numpy.float64(0)
        ave_testing_accuracy.P = numpy.float64(0)
        ave_testing_accuracy.F1 = numpy.float64(0)
        ave_testing_accuracy.auc = numpy.float64(0)
        for k in range(number_of_repeats):
            ave_testing_accuracy.overall_accuracy = ave_testing_accuracy.overall_accuracy + testing_accuracy[
                k].overall_accuracy / number_of_repeats
            ave_testing_accuracy.Gmean = ave_testing_accuracy.Gmean + testing_accuracy[k].Gmean / number_of_repeats
            ave_testing_accuracy.R = ave_testing_accuracy.R + testing_accuracy[k].R / number_of_repeats
            ave_testing_accuracy.P = ave_testing_accuracy.P + testing_accuracy[k].P / number_of_repeats
            ave_testing_accuracy.F1 = ave_testing_accuracy.F1 + testing_accuracy[k].F1 / number_of_repeats
            ave_testing_accuracy.auc = ave_testing_accuracy.auc + testing_accuracy[k].auc / number_of_repeats

        return testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy

    def my_loadmat_for_optimal_results(self, mat_path):
        my_results = scipy.io.loadmat(mat_path, mat_dtype=True, struct_as_record=False, squeeze_me=True)
        #  #struct_as_record=False, squeeze_me=True #simplify_cells=True #matlab_compatible=False
        # 这是我测试出来我觉着最好的结果

        """
        读进来的my_results包含"optimal_AE_optimizer_parameters","optimal_optimizer_parameters","optimal_mu",
        "optimal_lambda_error" (only for soft-boundary)
        "optimal_testing_accuracy","optimal_ave_testing_accuracy"

        optimal_AE_optimizer_parameters和optimal_optimizer_parameters是  My_Optimizer_Parameters类实例，包含以下变量：
        optimizer_name，
        learning_rate，
        learning_rate_milestones，
        number_of_epochs，
        batch_size，
        weight_decay，
        device，
        number_of_workers

        # optimal_AE_parameters是一个ndarray(number_of_AEs,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        # flag_GPU，int型
        # activation_function，str型
        # parameter_of_activation_function，numpy.float64型
        # number_of_hidden_layer_neurons，numpy.float64型
        # C，numpy.float64型

        optimal_testing_accuracy是一个ndarray(number_of_repeats,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        F1，numpy.float64型
        Gmean，numpy.float64型
        P，numpy.float64型
        R，numpy.float64型
        auc，numpy.float64型
        overall_accuracy，numpy.float64型

        optimal_ave_testing_accuracy是一个my_Struct结构体，包含以下变量：
        overall_accuracy，numpy.float64型
        Gmean，numpy.float64型
        R，numpy.float64型
        P，numpy.float64型
        F1，numpy.float64型
        auc，numpy.float64型
        """

        # region optimal_AE_optimizer_parameters和optimal_optimal_optimizer_parameters

        optimal_optimizer_parameters = My_Optimizer_Parameters(
            my_results['optimal_optimizer_parameters'].optimizer_name,
            my_results['optimal_optimizer_parameters'].learning_rate,
            my_results['optimal_optimizer_parameters'].learning_rate_milestones,
            my_results['optimal_optimizer_parameters'].number_of_epochs,
            my_results['optimal_optimizer_parameters'].batch_size,
            my_results['optimal_optimizer_parameters'].weight_decay,
            my_results['optimal_optimizer_parameters'].device,
            my_results['optimal_optimizer_parameters'].number_of_workers,
        )

        # endregion

        # region optimal_testing_accuracy
        optimal_testing_accuracy = numpy.array([])
        for k_testing_accuracy in my_results['optimal_testing_accuracy']:
            optimal_testing_accuracy = numpy.append(optimal_testing_accuracy, my_Struct())

            optimal_testing_accuracy[-1].overall_accuracy = numpy.float64(k_testing_accuracy.overall_accuracy)
            optimal_testing_accuracy[-1].Gmean = numpy.float64(k_testing_accuracy.Gmean)
            optimal_testing_accuracy[-1].R = numpy.float64(k_testing_accuracy.R)
            optimal_testing_accuracy[-1].P = numpy.float64(k_testing_accuracy.P)
            optimal_testing_accuracy[-1].F1 = numpy.float64(k_testing_accuracy.F1)
            optimal_testing_accuracy[-1].auc = numpy.float64(k_testing_accuracy.auc)
        # endregion

        # region ave_testing_accuracy
        optimal_ave_testing_accuracy = my_Struct()
        optimal_ave_testing_accuracy.overall_accuracy = numpy.float64(
            my_results['optimal_ave_testing_accuracy'].overall_accuracy)
        optimal_ave_testing_accuracy.Gmean = numpy.float64(my_results['optimal_ave_testing_accuracy'].Gmean)
        optimal_ave_testing_accuracy.R = numpy.float64(my_results['optimal_ave_testing_accuracy'].R)
        optimal_ave_testing_accuracy.P = numpy.float64(my_results['optimal_ave_testing_accuracy'].P)
        optimal_ave_testing_accuracy.F1 = numpy.float64(my_results['optimal_ave_testing_accuracy'].F1)
        optimal_ave_testing_accuracy.auc = numpy.float64(my_results['optimal_ave_testing_accuracy'].auc)
        # endregion

        # region mu, lambda_error (only for soft-boundary)
        optimal_scaling_parameter = my_results['optimal_scaling_parameter']
        optimal_mu = my_results['optimal_mu']
        optimal_warm_up_n_epochs = my_results['optimal_warm_up_n_epochs']
        optimal_number_of_hidden_neurons = my_results['optimal_number_of_hidden_neurons']

        return optimal_testing_accuracy, optimal_ave_testing_accuracy, \
               optimal_optimizer_parameters, optimal_number_of_hidden_neurons, optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs

        # endregion


class LBLSig_scaling_in_batch_OCITN(LBLSig_scaling_in_batch):

    def __init__(self, loss_function: str, net_name: str):

        super().__init__(loss_function, net_name)

    def build_network(self):

        # region build network
        if self.net_name in ('OCITN_CIFAR10', 'OCITN_SVHN_32'):
            self.net = OCITN_CIFAR10()

        else:
            raise SystemExit('Unknown Networks.')

        # endregion

    def compute_hypersphere_center(self, *args, **kwargs):

        if self.net_name in ('OCITN_CIFAR10', 'OCITN_SVHN_32'):
            I1 = Image.open('Lenna.png').resize([32, 32])

            trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            self.c = trans(I1).to(self.optimizer_parameters.device)

        else:
            raise SystemExit('Unknown Networks.')

    def run_with_optimal_parameters(self, save_name_results, train_dataset, test_dataset, number_of_repeats,
                                    flag_GPU: str, number_of_workers, flag_random_seed=True):

        optimal_testing_accuracy, optimal_ave_testing_accuracy, \
        optimal_optimizer_parameters, \
        optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs = \
            self.my_loadmat_for_optimal_results(
                os.path.join(save_name_results, 'optimal_results.mat'))

        optimal_optimizer_parameters.device = flag_GPU
        optimal_optimizer_parameters.number_of_workers = number_of_workers

        testing_accuracy = numpy.array([None] * number_of_repeats)
        ave_testing_accuracy = my_Struct()
        if flag_random_seed:
            set_random_seed(1)
        for k in range(number_of_repeats):
            self.build_network()
            self.train(train_dataset, optimal_optimizer_parameters, optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs)
            testing_accuracy[k] = self.test(test_dataset)

        ave_testing_accuracy.overall_accuracy = numpy.float64(0)
        ave_testing_accuracy.Gmean = numpy.float64(0)
        ave_testing_accuracy.R = numpy.float64(0)
        ave_testing_accuracy.P = numpy.float64(0)
        ave_testing_accuracy.F1 = numpy.float64(0)
        ave_testing_accuracy.auc = numpy.float64(0)
        for k in range(number_of_repeats):
            ave_testing_accuracy.overall_accuracy = ave_testing_accuracy.overall_accuracy + testing_accuracy[
                k].overall_accuracy / number_of_repeats
            ave_testing_accuracy.Gmean = ave_testing_accuracy.Gmean + testing_accuracy[k].Gmean / number_of_repeats
            ave_testing_accuracy.R = ave_testing_accuracy.R + testing_accuracy[k].R / number_of_repeats
            ave_testing_accuracy.P = ave_testing_accuracy.P + testing_accuracy[k].P / number_of_repeats
            ave_testing_accuracy.F1 = ave_testing_accuracy.F1 + testing_accuracy[k].F1 / number_of_repeats
            ave_testing_accuracy.auc = ave_testing_accuracy.auc + testing_accuracy[k].auc / number_of_repeats

        return testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy

    def my_loadmat_for_optimal_results(self, mat_path):
        my_results = scipy.io.loadmat(mat_path, mat_dtype=True, struct_as_record=False, squeeze_me=True)
        #  #struct_as_record=False, squeeze_me=True #simplify_cells=True #matlab_compatible=False
        # 这是我测试出来我觉着最好的结果

        """
        读进来的my_results包含"optimal_AE_optimizer_parameters","optimal_optimizer_parameters","optimal_mu",
        "optimal_lambda_error" (only for soft-boundary)
        "optimal_testing_accuracy","optimal_ave_testing_accuracy"

        optimal_AE_optimizer_parameters和optimal_optimizer_parameters是  My_Optimizer_Parameters类实例，包含以下变量：
        optimizer_name，
        learning_rate，
        learning_rate_milestones，
        number_of_epochs，
        batch_size，
        weight_decay，
        device，
        number_of_workers

        # optimal_AE_parameters是一个ndarray(number_of_AEs,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        # flag_GPU，int型
        # activation_function，str型
        # parameter_of_activation_function，numpy.float64型
        # number_of_hidden_layer_neurons，numpy.float64型
        # C，numpy.float64型

        optimal_testing_accuracy是一个ndarray(number_of_repeats,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        F1，numpy.float64型
        Gmean，numpy.float64型
        P，numpy.float64型
        R，numpy.float64型
        auc，numpy.float64型
        overall_accuracy，numpy.float64型

        optimal_ave_testing_accuracy是一个my_Struct结构体，包含以下变量：
        overall_accuracy，numpy.float64型
        Gmean，numpy.float64型
        R，numpy.float64型
        P，numpy.float64型
        F1，numpy.float64型
        auc，numpy.float64型
        """

        # region optimal_AE_optimizer_parameters和optimal_optimal_optimizer_parameters

        optimal_optimizer_parameters = My_Optimizer_Parameters(
            my_results['optimal_optimizer_parameters'].optimizer_name,
            my_results['optimal_optimizer_parameters'].learning_rate,
            my_results['optimal_optimizer_parameters'].learning_rate_milestones,
            my_results['optimal_optimizer_parameters'].number_of_epochs,
            my_results['optimal_optimizer_parameters'].batch_size,
            my_results['optimal_optimizer_parameters'].weight_decay,
            my_results['optimal_optimizer_parameters'].device,
            my_results['optimal_optimizer_parameters'].number_of_workers,
        )

        # endregion

        # region optimal_testing_accuracy
        optimal_testing_accuracy = numpy.array([])
        for k_testing_accuracy in my_results['optimal_testing_accuracy']:
            optimal_testing_accuracy = numpy.append(optimal_testing_accuracy, my_Struct())

            optimal_testing_accuracy[-1].overall_accuracy = numpy.float64(k_testing_accuracy.overall_accuracy)
            optimal_testing_accuracy[-1].Gmean = numpy.float64(k_testing_accuracy.Gmean)
            optimal_testing_accuracy[-1].R = numpy.float64(k_testing_accuracy.R)
            optimal_testing_accuracy[-1].P = numpy.float64(k_testing_accuracy.P)
            optimal_testing_accuracy[-1].F1 = numpy.float64(k_testing_accuracy.F1)
            optimal_testing_accuracy[-1].auc = numpy.float64(k_testing_accuracy.auc)
        # endregion

        # region ave_testing_accuracy
        optimal_ave_testing_accuracy = my_Struct()
        optimal_ave_testing_accuracy.overall_accuracy = numpy.float64(
            my_results['optimal_ave_testing_accuracy'].overall_accuracy)
        optimal_ave_testing_accuracy.Gmean = numpy.float64(my_results['optimal_ave_testing_accuracy'].Gmean)
        optimal_ave_testing_accuracy.R = numpy.float64(my_results['optimal_ave_testing_accuracy'].R)
        optimal_ave_testing_accuracy.P = numpy.float64(my_results['optimal_ave_testing_accuracy'].P)
        optimal_ave_testing_accuracy.F1 = numpy.float64(my_results['optimal_ave_testing_accuracy'].F1)
        optimal_ave_testing_accuracy.auc = numpy.float64(my_results['optimal_ave_testing_accuracy'].auc)
        # endregion

        # region mu, lambda_error (only for soft-boundary)
        optimal_scaling_parameter = my_results['optimal_scaling_parameter']
        optimal_mu = my_results['optimal_mu']
        optimal_warm_up_n_epochs = my_results['optimal_warm_up_n_epochs']

        return optimal_testing_accuracy, optimal_ave_testing_accuracy, \
               optimal_optimizer_parameters, optimal_scaling_parameter, optimal_mu, optimal_warm_up_n_epochs

        # endregion

from my_dataset import *
from my_customs import *
from LBL_networks import *
from LBLSig_scaling_in_batch_networks import *

import os
import numpy
import torch

import scipy.io
from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # region Basic parameter Setting

    number_of_repeats_run = 1

    number_of_repeats = 3
    number_of_epochs = 100 # 15 epoch and 32 batch size for PANDA-ResNet
    batch_size = 100
    flag_GPU = 'cuda'  #  'cuda' or 'cpu'
    number_of_workers = 0

    algorithm_type = 'LBL_OCITN'
    # LBL_DSLeNet, LBL_MLP, LBL_OCITN
    # LBLSig_scaling_in_batch_DSLeNet, LBLSig_scaling_in_batch_MLP, LBLSig_scaling_in_batch_OCITN

    net_name = 'OCITN_CIFAR10'
    # DSLeNet_CIFAR10, MLP_1_HL, MLP_2_HLs, OCITN_CIFAR10
    # DSLeNet_SVHN_32, OCITN_SVHN_32

    name_of_dataset = 'CIFAR10'
    dataset_type = 2
    # Dataset type
    # 1: OCC_datasets_matlab
    # 2: OCC_datasets_torch_CIFAR10
    # 3: OCC_datasets_torch_SVHN_32
    flag_normalization = 4
    # 5: For PANDA_ResNet

    evaluation_metric = 'auc' # overall_accuracy, Gmean, R, P, F1, auc

    class_chosen = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    class_chosen = numpy.array(class_chosen) - 1  # Subtract 1 because counting from 0 in Python

    path_of_dataset = os.path.join('.', 'datasets', name_of_dataset)

    # endregion

    class_testing_accuracy = [my_Struct() for i in range(len(class_chosen))]

    for index, k_number_of_classes in enumerate(class_chosen):

        save_name_results = os.path.join('.', 'results' + '-' + evaluation_metric)

        if dataset_type == 1:
            train_dataset = OCC_datasets_matlab(path_of_dataset, k_number_of_classes, flag_normalization, flag_train=True)
            test_dataset = OCC_datasets_matlab(path_of_dataset, k_number_of_classes, flag_normalization, flag_train=False)
            save_name_results = os.path.join(save_name_results,
                                             str(dataset_type)+'_'+name_of_dataset+'_'+str(flag_normalization),
                                             net_name)

        elif dataset_type == 2:
            train_dataset = OCC_datasets_torch_CIFAR10(path_of_dataset, k_number_of_classes, flag_normalization, flag_train=True)
            test_dataset = OCC_datasets_torch_CIFAR10(path_of_dataset, k_number_of_classes, flag_normalization, flag_train=False)
            save_name_results = os.path.join(save_name_results,
                                             str(dataset_type)+'_'+name_of_dataset+'_'+str(flag_normalization),
                                             net_name)

        elif dataset_type == 3:
            train_dataset = OCC_datasets_torch_SVHN_32(path_of_dataset, k_number_of_classes, flag_normalization, flag_train=True)
            test_dataset = OCC_datasets_torch_SVHN_32(path_of_dataset, k_number_of_classes, flag_normalization, flag_train=False)
            save_name_results = os.path.join(save_name_results,
                                             str(dataset_type)+'_'+name_of_dataset+'_'+str(flag_normalization),
                                             net_name)

        else:
            raise SystemError('Unknown dataset type.')

        if algorithm_type == 'LBL_DSLeNet':
            save_name_results_each_class = os.path.join(save_name_results,
                                                        'LBL' + '-' + str(number_of_repeats) + '-' + str(number_of_epochs) + '-' + str(batch_size),
                                                        'class_' + str(k_number_of_classes + 1))


            my_model = LBL_DSLeNet(loss_function='LBL', net_name=net_name)
            testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy = \
                my_model.run_with_optimal_parameters(save_name_results_each_class, train_dataset, test_dataset,
                                                     number_of_repeats_run,
                                                     flag_GPU, number_of_workers)

            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('loss_function=LBL')
            print('Net: {}, Dataset: {}, Class: {}'.format(net_name, name_of_dataset, k_number_of_classes))
            print('The optimal_ave_testing_accuracy.auc:{:.2%}'.format(optimal_ave_testing_accuracy.auc))
            print('The ave_testing_accuracy.auc:{:.2%}'.format(ave_testing_accuracy.auc))

        elif algorithm_type == 'LBL_MLP':
            save_name_results_each_class = os.path.join(save_name_results,
                                                        'LBL' + '-' + str(number_of_repeats) + '-' + str(number_of_epochs) + '-' + str(batch_size),
                                                        'class_' + str(k_number_of_classes + 1))


            my_model = LBL_MLP(loss_function='LBL', net_name=net_name)
            testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy = \
                my_model.run_with_optimal_parameters(save_name_results_each_class, train_dataset, test_dataset,
                                                     number_of_repeats_run,
                                                     flag_GPU, number_of_workers)

            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('loss_function=LBL')
            print('Net: {}, Dataset: {}, Class: {}'.format(net_name, name_of_dataset, k_number_of_classes))
            print('The optimal_ave_testing_accuracy.auc:{:.2%}'.format(optimal_ave_testing_accuracy.auc))
            print('The ave_testing_accuracy.auc:{:.2%}'.format(ave_testing_accuracy.auc))

        elif algorithm_type == 'LBL_OCITN':
            save_name_results_each_class = os.path.join(save_name_results,
                                                        'LBL' + '-' + str(number_of_repeats) + '-' + str(number_of_epochs) + '-' + str(batch_size),
                                                        'class_' + str(k_number_of_classes + 1))


            my_model = LBL_OCITN(loss_function='LBL', net_name=net_name)
            testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy = \
                my_model.run_with_optimal_parameters(save_name_results_each_class, train_dataset, test_dataset,
                                                     number_of_repeats_run,
                                                     flag_GPU, number_of_workers)

            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('loss_function=LBL')
            print('Net: {}, Dataset: {}, Class: {}'.format(net_name, name_of_dataset, k_number_of_classes))
            print('The optimal_ave_testing_accuracy.auc:{:.2%}'.format(optimal_ave_testing_accuracy.auc))
            print('The ave_testing_accuracy.auc:{:.2%}'.format(ave_testing_accuracy.auc))

        elif algorithm_type == 'LBLSig_scaling_in_batch_DSLeNet':
            save_name_results_each_class = os.path.join(save_name_results,
                                                        'LBLSig_scaling_in_batch' + '-' + str(number_of_repeats) + '-' + str(number_of_epochs) + '-' + str(batch_size),
                                                        'class_' + str(k_number_of_classes + 1))


            my_model = LBLSig_scaling_in_batch_DSLeNet(loss_function='LBLSig_scaling_in_batch', net_name=net_name)
            testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy = \
                my_model.run_with_optimal_parameters(save_name_results_each_class, train_dataset, test_dataset,
                                                     number_of_repeats_run,
                                                     flag_GPU, number_of_workers)

            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('loss_function=LBLSig_scaling_in_batch')
            print('Net: {}, Dataset: {}, Class: {}'.format(net_name, name_of_dataset, k_number_of_classes))
            print('The optimal_ave_testing_accuracy.auc:{:.2%}'.format(optimal_ave_testing_accuracy.auc))
            print('The ave_testing_accuracy.auc:{:.2%}'.format(ave_testing_accuracy.auc))

        elif algorithm_type == 'LBLSig_scaling_in_batch_MLP':
            save_name_results_each_class = os.path.join(save_name_results,
                                                        'LBLSig_scaling_in_batch' + '-' + str(number_of_repeats) + '-' + str(number_of_epochs) + '-' + str(batch_size),
                                                        'class_' + str(k_number_of_classes + 1))


            my_model = LBLSig_scaling_in_batch_MLP(loss_function='LBLSig_scaling_in_batch', net_name=net_name)
            testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy = \
                my_model.run_with_optimal_parameters(save_name_results_each_class, train_dataset, test_dataset,
                                                     number_of_repeats_run,
                                                     flag_GPU, number_of_workers)

            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('loss_function=LBLSig_scaling_in_batch')
            print('Net: {}, Dataset: {}, Class: {}'.format(net_name, name_of_dataset, k_number_of_classes))
            print('The optimal_ave_testing_accuracy.auc:{:.2%}'.format(optimal_ave_testing_accuracy.auc))
            print('The ave_testing_accuracy.auc:{:.2%}'.format(ave_testing_accuracy.auc))

        elif algorithm_type == 'LBLSig_scaling_in_batch_OCITN':
            save_name_results_each_class = os.path.join(save_name_results,
                                                        'LBLSig_scaling_in_batch' + '-' + str(number_of_repeats) + '-' + str(number_of_epochs) + '-' + str(batch_size),
                                                        'class_' + str(k_number_of_classes + 1))


            my_model = LBLSig_scaling_in_batch_OCITN(loss_function='LBLSig_scaling_in_batch', net_name=net_name)
            testing_accuracy, ave_testing_accuracy, optimal_testing_accuracy, optimal_ave_testing_accuracy = \
                my_model.run_with_optimal_parameters(save_name_results_each_class, train_dataset, test_dataset,
                                                     number_of_repeats_run,
                                                     flag_GPU, number_of_workers)

            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('loss_function=LBLSig_scaling_in_batch')
            print('Net: {}, Dataset: {}, Class: {}'.format(net_name, name_of_dataset, k_number_of_classes))
            print('The optimal_ave_testing_accuracy.auc:{:.2%}'.format(optimal_ave_testing_accuracy.auc))
            print('The ave_testing_accuracy.auc:{:.2%}'.format(ave_testing_accuracy.auc))

        else:
            raise SystemError('Unknown algorithm type.')

        class_testing_accuracy[index].class_chosen = k_number_of_classes
        class_testing_accuracy[index].testing_accuracy = testing_accuracy
        class_testing_accuracy[index].ave_testing_accuracy = ave_testing_accuracy
        class_testing_accuracy[index].optimal_testing_accuracy = optimal_testing_accuracy
        class_testing_accuracy[index].optimal_ave_testing_accuracy = optimal_ave_testing_accuracy

    for i in range(len(class_chosen)):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Class: {}'.format(class_testing_accuracy[i].class_chosen))
        print('Net: {}, Dataset: {}'.format(net_name, name_of_dataset))
        print('The optimal_ave_testing_accuracy.auc:{:.2%}'.format(class_testing_accuracy[i].optimal_ave_testing_accuracy.auc))
        print('The ave_testing_accuracy.auc:{:.2%}'.format(class_testing_accuracy[i].ave_testing_accuracy.auc))



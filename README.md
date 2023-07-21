# PyTorch Implementation of LBL and LBLSig

This repository provides a PyTorch implementations of the LBL and LBLSig methods presented in our paper 
<br>**"LBL: Logarithmic Barrier Loss Function for One-class Classification"**.
<br>You can find the original paper at <https://arxiv.org/abs/2307.10753>

## Citation

If you use our work, please also cite the paper:

	@misc{wang2023lbl,
		title={LBL: Logarithmic Barrier Loss Function for One-class Classification}, 
		author={Tianlei Wang and Dekang Liu and Wandong Zhang and Jiuwen Cao},
		year={2023},
		eprint={2307.10753},
		archivePrefix={arXiv},
		primaryClass={cs.CV}
	}

## Runing Environment

This program uses some common Python packages including Numpy, PyTorch, Scipy, Sklearn, Matplotlib. These packages can be installed easily by the Official Website description. I suggest using "conda" to conduct installation.

## Instruction

The folder "results-auc" saves the hyper-parameters that can reproduce the results listed in Table 1 of the paper. There may be deviations due to computer hardware differences even if set the same random seed.

For the MLP-type networks, we uses datasets with the ".mat" file format. For example, the results of Table 1 under the MLP backbone are obtained using the dataset "MATLAB_CIFAR10_colour". For this dataset, you should download the original CIFAR10 dataset first. Then, the MATLAB function in "./datasets/MATLAB_CIFAR10_colour/CIFAR10_my_formatting.m" can help you derive the dataset "MATLAB_CIFAR10_colour".

Directly Run "my_main_with_optimal_parameters.py", you can reproduce the results of "Table 1->OCITN->LBL", i.e.,

	algorithm_type = 'LBL_OCITN'
	net_name = 'OCITN_CIFAR10'
	name_of_dataset = 'CIFAR10'
	dataset_type = 2
	flag_normalization = 4

If you want to reproduce the results of "Table 1->OCITN->LBLSig", modify these variables of "my_main_with_optimal_parameters.py" as 
	
	algorithm_type = 'LBLSig_scaling_in_batch_OCITN'
	net_name = 'OCITN_CIFAR10'
	name_of_dataset = 'CIFAR10'
	dataset_type = 2
	flag_normalization = 4

If you want to reproduce the results of "Table 1->MLP->LBL or LBLSig", modify these variables of "my_main_with_optimal_parameters.py" as 

	algorithm_type = 'LBL_MLP' or 'LBLSig_scaling_in_batch_MLP'
	net_name = 'MLP_2_HLs'
	name_of_dataset = 'MATLAB_CIFAR10_colour'
	dataset_type = 1
	flag_normalization = 1.1

If you want to reproduce the results of "Table 1->DSLeNet->LBL or LBLSig", modify these variables of "my_main_with_optimal_parameters.py" as 

	algorithm_type = 'LBL_DSLeNet' or 'LBLSig_scaling_in_batch_DSLeNet'
	net_name = 'DSLeNet_CIFAR10'
	name_of_dataset = 'CIFAR10'
	dataset_type = 2
	flag_normalization = 4

## License
MIT




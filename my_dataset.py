from my_customs import *
import os
import numpy
import torch
import scipy.io
from sklearn import preprocessing
import torchvision

# from torch.utils.data import Subset

from PIL import Image


# 需要说明，不在数据集类中建立Dataloader是因为建立的数据集是通用的
# Dataloader更偏向于重复操作，没必要重复进行导致代码量增加

# self.label: 保存的是完整数据集，各个类的label
# self.target_label: 目标类的label
# self.outlier_label: 非目标类的label
# self.data: 数据
# self.targets: 数据对应的label

class OCC_datasets_matlab(torch.utils.data.Dataset):
    def __init__(self, path_of_dataset: str, class_chosen, flag_normalization, flag_train: bool):
        # class_chosen是从0开始计数的

        super().__init__()

        self.flag_normalization = flag_normalization
        self.flag_train = flag_train

        # region 确定有多少类，以及目标类的label和非目标类的label
        T = scipy.io.loadmat(os.path.join(path_of_dataset, 'train_label.mat'), mat_dtype=True)
        # T = torch.DoubleTensor(T['T'])
        T = T['T']

        # label = torch.unique(T) # 确定有什么类
        self.label = numpy.unique(T)
        self.number_of_classes = len(self.label)  # 确定有多少类
        del T
        if (-1) in self.label:
            # 判断类号里面有没有-1，因为下面分数据集的时候，是把非目标类的类号置为 - 1，而且在程序中，并没有把目标类的类号全部置为1，而是依然保留原始的类号
            # 因此，如果目标类的类号是-1的话，那么最后的预测结果都是-1，准确率就是100%
            # print('The label includes "-1"')
            raise SystemExit('The label includes "-1"')

        self.target_label = self.label[class_chosen]  # 目标类
        self.outlier_label = self.label.tolist()
        self.outlier_label.remove(self.target_label)
        self.outlier_label = numpy.array(self.outlier_label)  # 非目标类

        # endregion

        ## 处理数据集
        # 这里不用torch.DoubleTensor进行转换，
        # 是因为下面的归一化preprocessing.MinMaxScaler还是需要ndarray。
        # 即使转化了，在归一化后还是会恢复成ndarray格式。

        # region 载入数据集
        if flag_train:
            P = scipy.io.loadmat(os.path.join(path_of_dataset, 'train_data.mat'), mat_dtype=True)
            self.data = P['P']

            T = scipy.io.loadmat(os.path.join(path_of_dataset, 'train_label.mat'), mat_dtype=True)
            self.targets = T['T']
            # self.targets是(N,1)形状的数组

            del P
            del T

        else:
            TV_P = scipy.io.loadmat(os.path.join(path_of_dataset, 'test_data.mat'), mat_dtype=True)
            self.data = TV_P['TV_P']

            TV_T = scipy.io.loadmat(os.path.join(path_of_dataset, 'test_label.mat'), mat_dtype=True)
            self.targets = TV_T['TV_T']

            del TV_P
            del TV_T
        # endregion

        # region 归一化
        if flag_normalization == 0:
            print('Without Normalization')

        elif flag_normalization == 1:  # 对灰白图像进行整体归一化
            self.data = self.data / 255

        elif flag_normalization == 1.1:
            self.data = self.data * 2 - 1

        elif flag_normalization == 2:
            # preprocessing.MinMaxScaler()是对每一列进行归一化
            if flag_train:
                ps = preprocessing.MinMaxScaler((-1, 1))
                self.data = ps.fit_transform(self.data.T).T

            else:
                P = scipy.io.loadmat(os.path.join(path_of_dataset, 'train_data.mat'), mat_dtype=True)
                P = P['P']

                ps = preprocessing.MinMaxScaler((-1, 1))
                P = ps.fit_transform(P.T).T
                self.data = ps.transform(self.data.T).T
                del P

            del ps

        elif flag_normalization == 3:  # 对整个数据集进行统一的归一化
            self.data = ((self.data - self.data.min()) / (self.data.max() - self.data.min())) * 2 - 1

        elif flag_normalization == 4:
            # 因为使用HRN时，gradient_penalty很大，很容易出现NAN或者INF。
            # 参考了下源代码，发现对数据额外做了一个处理
            # 源代码里面做的处理，看起来很像global contrast normalization (GCN)，但是又不一样
            # 源代码里面是先用了L2归一化，然后在减去均值。此外，标准GCN并不是用L2，而是平方的均值
            # 不知道作者是出于什么样的原因，但是我们还是采用标准的GCN进行归一化
            # nice，结局一样
            # 当初始学习率是1e-1，第1个epoch的第1个batch，gradient_penalty=4.8976e+13，然后第2个batch就INF了。
            # 当初始学习率是1e-2，第1个epoch的第1个batch，gradient_penalty=4.8976e+13，然后第2个batch，gradient_penalty=1.5032e+34，然后基本稳定在1e34的数量级，一样的结果
            # 那试试看用L2范数进行归一化了
            # 后续在flag_normalization == 4.2
            GCN_lambda = 0
            GCN_epsilon = 1e-8
            s = 1

            data_mean = numpy.mean(self.data, axis=0, keepdims=True)
            self.data = self.data - data_mean

            contrast = numpy.sqrt(GCN_lambda + numpy.mean(self.data ** 2, axis=0, keepdims=True))

            self.data = s * self.data / numpy.maximum(contrast, GCN_epsilon * numpy.ones_like(contrast))

        elif flag_normalization == 4.2:
            # 呵呵，并没有什么卵用。
            # 我发现源代码的网络输出很小，甚至最大值都到了1e-8或者1e-9的数量级了
            # 这可能是源代码中随机初始化的值，以及源代码中使用的是单个输出节点导致的？
            # 后续返回到损失函数那里
            GCN_lambda = 0
            GCN_epsilon = 1e-8
            s = 1

            data_mean = numpy.mean(self.data, axis=0, keepdims=True)
            self.data = self.data - data_mean

            contrast = numpy.sqrt(GCN_lambda + numpy.sum(self.data ** 2, axis=0, keepdims=True))

            self.data = s * self.data / numpy.maximum(contrast, GCN_epsilon * numpy.ones_like(contrast))

        else:
            # print("unknow case.")
            raise SystemExit('Unknow switch case.')

        # endregion

        # region 生成单分类数据集
        if flag_train:
            target_positions = numpy.argwhere(self.targets == self.target_label)  # 目标类的位置
            outlier_positions = numpy.argwhere(self.targets != self.target_label)  # 非目标类的位置
            self.targets = self.targets[target_positions[:, 0]]
            self.data = self.data[:, target_positions[:, 0]]
            del target_positions
            del outlier_positions

        else:
            target_positions = numpy.argwhere(self.targets == self.target_label)  # 目标类的位置
            outlier_positions = numpy.argwhere(self.targets != self.target_label)  # 非目标类的位置
            self.targets[outlier_positions[:, 0]] = -1
            # 这时候这里是一个(N, 1)的二维数组，当使用TV_T[n]这种方式，提取的是第n行的元素（是一个（1，）的一维数组）。
            # 但是对TV_T[n]进行赋值时，会对第n行的每一个元素赋相同的值。
            del target_positions
            del outlier_positions

        # endregion

        self.targets = self.targets.flatten()
        # 和下面的CIFAR10统一，以及以后所有数据集都统一，用flatten变成(N,)的形状

        self.number_of_data = self.data.shape[1]
        self.number_of_data_outlier_class = numpy.argwhere(self.targets == -1).shape[
            0]  # numpy.argwhere得到的矩阵，每一行对应的一个元素的行指标和列指标
        self.number_of_data_target_class = self.number_of_data - self.number_of_data_outlier_class

    def __getitem__(self, item):
        # 本来返回值中不想放入target，但是因为考虑到建立Dataloader的时候，shuffle的影响，为了减少出错的可能性，还是同时返回了target
        # 在Dataloader中，返回的data：每一行是一个样本
        return torch.from_numpy(self.data[:, item]).float(), self.targets[item]

    def __len__(self):
        return self.data.shape[1]


class OCC_datasets_torch_CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, path_of_dataset: str, class_chosen, flag_normalization, flag_train: bool):

        # CIFAR10的label是从0开始的，对应关系如下：
        # 0: airplane; 1: automobile; 2: bird; 3: cat; 4: deer; 5: dog; 6: frog; 7: horse; 8: ship; 9: truck
        # 所以注意class_chosen的取值

        self.flag_train = flag_train

        # region 确定有多少类，以及目标类的label和非目标类的label
        self.target_label = numpy.array(class_chosen)  # 目标类
        self.outlier_label = list(range(0, 10))
        self.outlier_label.remove(self.target_label)
        self.outlier_label = numpy.array(self.outlier_label)  # 非目标类

        # endregion

        # region 构建transform
        if flag_normalization == 0:
            print('Without Normalization')

        elif flag_normalization == 1:
            my_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            # 因为我下面统一做处理了，所以这里并不需要target_transform了。而且在Deep SVDD中，target_transform的作用也就是为了最后计算AUC
            # my_target_transform = torchvision.transforms.Lambda(lambda x: int(x in self.target_label))
            # 把目标类的标签设为1，非目标类的标签设为0

        elif flag_normalization == 4:
            # 因为使用HRN时，gradient_penalty很大，很容易出现NAN。
            # 参考了下源代码，发现对数据额外做了一个处理
            # 源代码里面做的处理，看起来很像global contrast normalization (GCN)，但是又不一样
            # 源代码里面是先用了L2归一化，然后在减去均值。此外，标准GCN并不是用L2，而是平方的均值
            # 不知道作者是出于什么样的原因，但是我们还是采用标准的GCN进行归一化
            # nice，还是不对，还是无穷大
            my_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Lambda(
                                                               lambda x: global_contrast_normalization(x)
                                                           )
                                                           ])

        elif flag_normalization == 5:  # 用于PANDA_ResNet
            my_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                           torchvision.transforms.CenterCrop(224),
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224, 0.225])])

        else:
            raise SystemExit('Unknow switch case.')

        # endregion

        # region 继承torchvision.datasets.CIFAR10类
        super().__init__(root=path_of_dataset, download=True,
                         train=flag_train, transform=my_transform)

        # endregion

        # region 生成单分类数据集
        # 需要注意的是，继承的torchvision.datasets.CIFAR10类，数据分别被保存到self.data和self.targets中
        # self.data是numpy.ndarray格式(50000,32,32,3)，dtype=uint8；self.target是list格式
        self.data = numpy.array(self.data)
        self.targets = numpy.array(self.targets)  # 这里返回的是(N, )形状
        if flag_train:
            target_positions = numpy.argwhere(self.targets == self.target_label)  # 目标类的位置
            outlier_positions = numpy.argwhere(self.targets != self.target_label)  # 非目标类的位置
            self.targets = self.targets[target_positions[:, 0]]
            self.data = self.data[target_positions[:, 0], :, :, :]
            del target_positions
            del outlier_positions

        else:
            target_positions = numpy.argwhere(self.targets == self.target_label)  # 目标类的位置
            outlier_positions = numpy.argwhere(self.targets != self.target_label)  # 非目标类的位置
            self.targets[outlier_positions[:, 0]] = -1
            del target_positions
            del outlier_positions

        # endregion

        self.number_of_data = len(self.data)
        self.number_of_data_outlier_class = numpy.argwhere(self.targets == -1).shape[
            0]  # numpy.argwhere得到的矩阵，每一行对应的一个元素的行指标和列指标
        self.number_of_data_target_class = self.number_of_data - self.number_of_data_outlier_class

    def __getitem__(self, index):
        """original __getitem__ returns (image, target) where target is index of the target class."""

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # 本来返回值中不想放入target，但是因为考虑到建立Dataloader的时候，shuffle的影响，为了减少出错的可能性，还是同时返回了target
        # 需要说明的是，self.targets是numpy.ndarray；self.targets[index]是一个值，在这里是numpy.int32。
        # 在Dataloader中，经过不断选取，最后返回的是一个(N,)的一维张量Tensor型。
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], self.targets[index]
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img)
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     return img, target
    #
    # def __len__(self) -> int:
    #     return len(self.data)


class OCC_datasets_torch_SVHN_32(torchvision.datasets.SVHN):

    def __init__(self, path_of_dataset: str, class_chosen, flag_normalization, flag_train: bool):
        """
        SVHN Dataset. Note: The SVHN dataset assigns the label 10 to the digit 0.
        However, in this Dataset, we assign the label 0 to the digit 0 to be compatible with PyTorch loss functions
        which expect the class labels to be in the range [0, C-1].
        所以PyTorch中的SVHN的label是从0开始的，对应关系如下：
        0: 数字0; 1: 数字1; 2: 数字2; 3: 数字3; 4: 数字4; 5: 数字5; 6: 数字6; 7: 数字7; 8: 数字8; 9: 数字9
        所以注意class_chosen的取值
        """
        if flag_train == 1:
            self.split = 'train'
        else:
            self.split = 'test'


        # region 确定有多少类，以及目标类的label和非目标类的label
        self.target_label = numpy.array(class_chosen)  # 目标类
        self.outlier_label = list(range(0, 10))
        self.outlier_label.remove(self.target_label)
        self.outlier_label = numpy.array(self.outlier_label)  # 非目标类

        # endregion

        # region 构建transform
        if flag_normalization == 0:
            print('Without Normalization')

        elif flag_normalization == 1:
            my_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            # 因为我下面统一做处理了，所以这里并不需要target_transform了。而且在Deep SVDD中，target_transform的作用也就是为了最后计算AUC
            # my_target_transform = torchvision.transforms.Lambda(lambda x: int(x in self.target_label))
            # 把目标类的标签设为1，非目标类的标签设为0

        elif flag_normalization == 4:
            my_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Lambda(
                                                               lambda x: global_contrast_normalization(x)
                                                           )
                                                           ])

        else:
            raise SystemExit('Unknow switch case.')

        # endregion

        # region 继承torchvision.datasets.SVHN类
        super().__init__(root=path_of_dataset, download=True,
                         split=self.split, transform=my_transform)

        # endregion

        # region 生成单分类数据集
        # 需要注意的是，继承的torchvision.datasets.SVHN类，数据分别被保存到self.data和self.labels中
        # self.data是numpy.ndarray格式(73257,3,32,32)，dtype=uint8；
        # self.labels是numpy.ndarray格式(73257,)，dtype=int64；
        if flag_train:
            target_positions = numpy.argwhere(self.labels == self.target_label)  # 目标类的位置
            outlier_positions = numpy.argwhere(self.labels != self.target_label)  # 非目标类的位置
            self.labels = self.labels[target_positions[:, 0]]
            self.data = self.data[target_positions[:, 0], :, :, :]
            del target_positions
            del outlier_positions

        else:
            target_positions = numpy.argwhere(self.labels == self.target_label)  # 目标类的位置
            outlier_positions = numpy.argwhere(self.labels != self.target_label)  # 非目标类的位置
            self.labels[outlier_positions[:, 0]] = -1
            del target_positions
            del outlier_positions

        # endregion

        self.number_of_data = len(self.data)
        self.number_of_data_outlier_class = numpy.argwhere(self.labels == -1).shape[
            0]  # numpy.argwhere得到的矩阵，每一行对应的一个元素的行指标和列指标
        self.number_of_data_target_class = self.number_of_data - self.number_of_data_outlier_class

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(numpy.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # 在Dataloader中，经过不断选取，最后返回的是一个(N,)的一维张量Tensor型。
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], int(self.labels[index])
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(np.transpose(img, (1, 2, 0)))
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     return img, target
    #
    #
    # def __len__(self) -> int:
    #     return len(self.data)

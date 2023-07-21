from sklearn import metrics
import scipy.io
import numpy
import torch

class Dataset():

    def __init__(self, P, T, TV_P, TV_T):
        self.P = P
        self.T = T
        self.TV_P = TV_P
        self.TV_T = TV_T

class my_Struct():
    pass

class My_Optimizer_Parameters():
    def __init__(self, optimizer_name: str, learning_rate, learning_rate_milestones,
                 number_of_epochs, batch_size, weight_decay,
                 device, number_of_workers):
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.learning_rate_milestones = learning_rate_milestones
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.number_of_workers = number_of_workers


def my_roc_curve(deci,label_y,label_of_positive_samples,type_of_deci):
    # deci is the actual output
    # label_y is the desired output, i.e., the true label. It
    # label_of_positive_samples is the label of the positive_samples
    # type_of_deci is the actual output type, including
    #   1: the probability belonging to positive sample.
    #   2: the distance. The smaller distance, the closer to positive sample.
    if type_of_deci == 1:
        pass # 属于正样本的概率值从大到小排序
    elif type_of_deci == 2:
        deci = 1/deci
    elif type_of_deci == 3:
        deci = -deci # 因为用KNN的时候会出现等于0的情况（难以置信），所以加了取负
    else:
        raise SystemExit('Unknown switch case.')

    fpr, tpr, thresholds = metrics.roc_curve(label_y.flatten(), deci.flatten(), pos_label=label_of_positive_samples)
    auc = metrics.auc(fpr, tpr)
    return auc


def set_random_seed(seed):
    """
    虽然我设置了同样的种子，但是我发现在不同的电脑上，得到的结果依然不一致
    虽然不同电脑，同一个随机种子，使用torch.rand(100)得到的结果是一样的。╮(╯▽╰)╭
    """
    if seed is not None:
        # ---- set random seed
        # random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def global_contrast_normalization(x: torch.tensor, s=1, GCN_lambda=0, GCN_epsilon=1e-8):
    x_mean = torch.mean(x)

    x = x - x_mean

    contrast = torch.sqrt(GCN_lambda + torch.mean(x ** 2))

    x = s * x / torch.max(contrast, torch.tensor(GCN_epsilon))

    return x

import torch
import torch.nn as t_nn
import torch.nn.functional as t_nn_f


class OCITN_CIFAR10(t_nn.Module):

    def __init__(self):
        super().__init__()

        self.conv2d_1 = t_nn.Conv2d(3, 32, 3, padding='same')

        self.conv2d_2_to_7 = t_nn.ModuleList([t_nn.Conv2d(32, 32, 3, padding='same') for i in range(6)])

        self.conv2d_8 = t_nn.Conv2d(32, 3, 3, padding='same')

    def forward(self, x):
        x = t_nn_f.relu( self.conv2d_1(x) )

        for k_conv2d in self.conv2d_2_to_7:
            x = t_nn_f.relu( k_conv2d(x) )

        x = t_nn_f.relu( self.conv2d_8(x) )

        return x


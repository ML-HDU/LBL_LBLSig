import torch
import torch.nn as t_nn
import torch.nn.functional as t_nn_f




#  Deep SVDD论文搭建的LeNet型网络，针对CIFAR10
class DSLeNet_CIFAR10(t_nn.Module):

    def __init__(self):
        super().__init__()

        self.output_dimension = 128
        self.pool = t_nn.MaxPool2d(2, 2)

        self.conv2d_1 = t_nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d_1 = t_nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2d_2 = t_nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d_2 = t_nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv2d_3 = t_nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d_3 = t_nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc_1 = t_nn.Linear(128 * 4 * 4, self.output_dimension, bias=False)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.pool(t_nn_f.leaky_relu(self.bn2d_1(x)))
        x = self.conv2d_2(x)
        x = self.pool(t_nn_f.leaky_relu(self.bn2d_2(x)))
        x = self.conv2d_3(x)
        x = self.pool(t_nn_f.leaky_relu(self.bn2d_3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        return x


class DSLeNet_CIFAR10_Autoencoder(t_nn.Module):

    def __init__(self):
        super().__init__()

        self.output_dimension = 128
        self.pool = t_nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv2d_1 = t_nn.Conv2d(3, 32, 5, bias=False, padding=2)
        t_nn.init.xavier_uniform_(self.conv2d_1.weight, gain=t_nn.init.calculate_gain('leaky_relu'))
        # https://pytorch.org/docs/stable/nn.init.html
        self.bn2d_1 = t_nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2d_2 = t_nn.Conv2d(32, 64, 5, bias=False, padding=2)
        t_nn.init.xavier_uniform_(self.conv2d_2.weight, gain=t_nn.init.calculate_gain('leaky_relu'))
        self.bn2d_2 = t_nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv2d_3 = t_nn.Conv2d(64, 128, 5, bias=False, padding=2)
        t_nn.init.xavier_uniform_(self.conv2d_3.weight, gain=t_nn.init.calculate_gain('leaky_relu'))
        self.bn2d_3 = t_nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc_1 = t_nn.Linear(128 * 4 * 4, self.output_dimension, bias=False)
        self.bn1d = t_nn.BatchNorm1d(self.output_dimension, eps=1e-04, affine=False)

        # Decoder
        self.deconv2d_1 = t_nn.ConvTranspose2d(int(self.output_dimension / (4 * 4)), 128, 5, bias=False, padding=2)
        t_nn.init.xavier_uniform_(self.deconv2d_1.weight, gain=t_nn.init.calculate_gain('leaky_relu'))
        self.bn2d_4 = t_nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2d_2 = t_nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        t_nn.init.xavier_uniform_(self.deconv2d_2.weight, gain=t_nn.init.calculate_gain('leaky_relu'))
        self.bn2d_5 = t_nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv2d_3 = t_nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        t_nn.init.xavier_uniform_(self.deconv2d_3.weight, gain=t_nn.init.calculate_gain('leaky_relu'))
        self.bn2d_6 = t_nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv2d_4 = t_nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        t_nn.init.xavier_uniform_(self.deconv2d_4.weight, gain=t_nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.pool(t_nn_f.leaky_relu(self.bn2d_1(x)))
        x = self.conv2d_2(x)
        x = self.pool(t_nn_f.leaky_relu(self.bn2d_2(x)))
        x = self.conv2d_3(x)
        x = self.pool(t_nn_f.leaky_relu(self.bn2d_3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc_1(x))
        x = x.view(x.size(0), int(self.output_dimension / (4 * 4)), 4, 4)
        x = t_nn_f.leaky_relu(x)
        x = self.deconv2d_1(x)
        x = t_nn_f.interpolate(t_nn_f.leaky_relu(self.bn2d_4(x)), scale_factor=2)
        x = self.deconv2d_2(x)
        x = t_nn_f.interpolate(t_nn_f.leaky_relu(self.bn2d_5(x)), scale_factor=2)
        x = self.deconv2d_3(x)
        x = t_nn_f.interpolate(t_nn_f.leaky_relu(self.bn2d_6(x)), scale_factor=2)
        x = self.deconv2d_4(x)
        x = torch.sigmoid(x)
        return x

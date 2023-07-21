import torch
import torch.nn as t_nn
import torch.nn.functional as t_nn_f


# 这里没有用HRN论文中的MLP网络，是因为
# 对于CIFAR10的网络结构，在输入层分别对3个通道单独做了映射后进行拼接，这种结构又要对数据进行特别处理，觉着麻烦
# 所以弃而不用，选择一般更通用的

class MLP_1_HL(t_nn.Module):

    def __init__(self, number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons):
        # 因为HRN的输出层是1个神经元，所以这里还是留下来一个number_of_output_neurons接口
        super().__init__()
        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons

        self.in2hl = torch.nn.Linear(number_of_input_neurons, number_of_hidden_neurons)
        self.hl2ou = torch.nn.Linear(number_of_hidden_neurons, number_of_output_neurons)

    def forward(self, x):
        x = t_nn_f.leaky_relu(self.in2hl(x))
        x = self.hl2ou(x)
        return x


class MLP_2_HLs(t_nn.Module):

    def __init__(self, number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons):
        # 因为HRN的输出层是1个神经元，所以这里还是留下来一个number_of_output_neurons接口
        super().__init__()
        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons

        self.in2hl1 = torch.nn.Linear(number_of_input_neurons, number_of_hidden_neurons[0])
        self.hl12hl2 = torch.nn.Linear(number_of_hidden_neurons[0], number_of_hidden_neurons[1])
        self.hl22ou = torch.nn.Linear(number_of_hidden_neurons[1], number_of_output_neurons)

    def forward(self, x):
        x = t_nn_f.leaky_relu(self.in2hl1(x))
        x = t_nn_f.leaky_relu(self.hl12hl2(x))
        x = self.hl22ou(x)
        return x



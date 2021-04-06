import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class TimeBlock(nn.Module, ABC):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """
    # knernel 7 is bad, 3 is okay
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, kernel_size),
                               padding=(0, 1), padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, kernel_size),
                               padding=(0, 1), padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, kernel_size),
                               padding=(0, 1), padding_mode='zeros')

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes,
                                       num_timesteps,
                                       num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
                                       num_timesteps_out,
                                       num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)

        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))             # We can mimic this way to have residual model
        """
        temp = temp.permute(0, 2, 3, 1)
        X = X.permute(0, 2, 3, 1)
        X = torch.repeat_interleave(X, int(temp.shape[3] / X.shape[3]), dim=3)
        in_tensor = temp + X
        in_tensor = in_tensor.permute(0, 3, 1, 2)
        out = F.relu(in_tensor)
        """
        out = F.relu(out)
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)

        return out


class STGCNBlock(nn.Module, ABC):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """

        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])  # node, sample, time, feature
        # t2 = F.leaky_relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        # return self.batch_norm(t3)
        return t3


class STGCN_Autoencoder(nn.Module, ABC):
    """
       Autoencoder with spatio-temporal graph convolutional layers.
    """
    def __init__(self, num_nodes, in_feat, size_step, time_step):
        """
        :param num_nodes: Number of nodes in the graph
        :param in_feat: Number of features at each node in each time step
        :param size_step: Adjustable dimension of features at each layer
        :param time_step: Number of time steps fed into the network
        """
        super(STGCN_Autoencoder, self).__init__()
        self.time_step = time_step
        out_feat = in_feat
        spatial_channel = 16  # number of features after graph convolution
        out1, out2 = 300, 360
        self.batch_norm1 = nn.BatchNorm2d(num_nodes)
        self.batch_norm2 = nn.BatchNorm2d(num_nodes)
        self.batch_norm3 = nn.BatchNorm2d(num_nodes)

        self.block1 = STGCNBlock(in_channels=in_feat, spatial_channels=in_feat + 1 * size_step,
                                 out_channels=in_feat + 1*size_step, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=in_feat + 1 * size_step, spatial_channels=in_feat + 1 * size_step,
                                 out_channels=in_feat + 1 * size_step, num_nodes=num_nodes)

        self.conv1 = nn.Conv2d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.ConvTranspose2d(in_channels=out2, out_channels=out1, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=out1, out_channels=num_nodes, kernel_size=3, stride=1, padding=1)

        self.last_temporal = TimeBlock(in_channels=in_feat + 1 * size_step, out_channels=out_feat + 1 * size_step)
        self.linear_nn = nn.Linear(in_feat + 1 * size_step, out_feat)

    def forward(self, A_hat, X):
        """
        :param A_hat: Normalized adjacency matrix.
        :param X: Input data of shape (batch_size, num_nodes,
                                       num_timesteps, num_features).
        """
        h1 = X

        h1 = self.block1(h1, A_hat)
        h1 = self.batch_norm3(h1)
        h2 = self.block2(h1, A_hat)
        h2 = self.batch_norm3(h2)

        """
        h2 = self.conv1(h2)
        h2 = self.batch_norm2(h2)
        h2 = F.leaky_relu(h2)
        """

        h3 = self.last_temporal(h2)
        imputed_data = torch.sigmoid(self.linear_nn(h3))
        # imputed_data = torch.sigmoid(self.last_temporal(h2))
        # imputed_data = self.linear_nn2(h3)

        return imputed_data




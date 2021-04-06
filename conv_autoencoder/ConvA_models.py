import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class ConvAutoencoderModel(nn.Module, ABC):
    """
        Autoencoder with 1-D convolutional layers.
    """
    def __init__(self, num_nodes, feat_num):
        """
        :param num_nodes: Number of nodes in the graph.
        :param feat_num: Number of features at each node in each time step.
        """
        super(ConvAutoencoderModel, self).__init__()
        super().__init__()

        out1, out2 = 300, 360
        self.batch_norm1 = nn.BatchNorm1d(out1)
        self.batch_norm2 = nn.BatchNorm1d(out2)

        # Encoder out1 = 360, out2 = 480

        self.conv1 = nn.Conv1d(in_channels=num_nodes, out_channels=out1, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=out2, out_channels=out2, kernel_size=5, stride=1, padding=2)

        # Fully connected layer
        self.linear_nn = nn.Linear(feat_num, feat_num)

        # Decoder
        self.up1 = nn.ConvTranspose1d(in_channels=out2, out_channels=out2, kernel_size=5, stride=1, padding=2)
        self.up2 = nn.ConvTranspose1d(in_channels=out2, out_channels=out1, kernel_size=5, stride=1, padding=2)
        self.up3 = nn.ConvTranspose1d(in_channels=out1, out_channels=num_nodes,
                                      kernel_size=5, stride=1, padding=2)

    def encoder(self, X):
        conv1 = self.conv1(X)  # torch.Size([25, 300, 48])
        conv1 = self.batch_norm1(conv1)  # torch.Size([25, 300, 48])
        relu1 = F.leaky_relu(conv1)  # torch.Size([25, 300, 48])
        conv2 = self.conv2(relu1)  # torch.Size([25, 360, 48])
        conv2 = self.batch_norm2(conv2)  # torch.Size([25, 360, 48])
        relu2 = F.leaky_relu(conv2)  # torch.Size([25, 360, 48])
        conv3 = self.conv3(relu2)  # torch.Size([25, 360, 48])
        conv3 = self.batch_norm2(conv3)  # torch.Size([25, 360, 48])
        relu3 = F.leaky_relu(conv3)  # torch.Size([25, 360, 48])
        return relu3

    def fully_connected(self, encoding):
        X_mid = self.linear_nn(encoding)  # torch.Size([25, 360, 48])
        return X_mid

    def decoder(self, X_mid):
        up1 = self.up1(X_mid)  # torch.Size([25, 360, 48])
        up1 = self.batch_norm2(up1)  # torch.Size([25, 360, 48])
        up_relu1 = F.leaky_relu(up1)  # torch.Size([25, 360, 48])
        up2 = self.up2(up_relu1)  # torch.Size([25, 300, 48])
        up2 = self.batch_norm1(up2)  # torch.Size([25, 300, 48])
        up_relu2 = F.leaky_relu(up2)  # torch.Size([25, 300, 48])
        up3 = self.up3(up_relu2)  # torch.Size([25, 240, 48])
        imputed = torch.sigmoid(up3)  # torch.Size([25, 240, 48])

        return imputed

    def forward(self, X):
        # X = self.drop_out(X)
        encoding = self.encoder(X)
        X_mid = self.fully_connected(encoding)
        imputed = self.decoder(X_mid)
        return imputed



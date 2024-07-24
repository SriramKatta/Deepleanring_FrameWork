import torch.nn as nn
# from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.batch_dim = None

    def forward(self, input_tensor):
        self.batch_dim = input_tensor.shape[0]
        return input_tensor.reshape(self.batch_dim, -1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_shape=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride_shape, 1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.residual_conv = True
        self.conv1X1 = nn.Conv2d(in_channels, out_channels, 1, stride_shape)
        if in_channels == out_channels and stride_shape == 1:
            self.residual_conv = False
        else:
            self.residual_conv = True

        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu_3 = nn.ReLU()
        self.seq = nn.Sequential(self.conv1, self.batch_norm1, self.relu_1, self.conv2, self.batch_norm2)
        self.residual = None
        self.cnt = 0

    def forward(self, input_tensor):
        self.residual = input_tensor
        output_tensor = self.seq(input_tensor)
        if self.residual_conv:
            self.residual = self.conv1X1(self.residual)
        self.residual = self.batch_norm3(self.residual)
        output_tensor += self.residual
        output_tensor = self.relu_3(output_tensor)
        return output_tensor
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.seq2 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            nn.Dropout(),
            ResBlock(256, 512, 2)
        )

        self.seq3 = nn.Sequential(
            nn.AvgPool2d(10),
            Flatten(),
            nn.Dropout(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        return self.seq3(self.seq2(self.seq1(input_tensor)))
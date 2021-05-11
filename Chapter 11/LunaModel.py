import torch.nn as nn 
import torch.utils.data.dataloader as dataloader
import math

# class LunaBlock(nn.Module):
#     def __init__(self, in_channels, conv_channels):
#         super().__init__()

#         self.in_channels = in_channels
#         self.conv_channels = conv_channels

#         # kernelsize =3 is actually (3,3,3) for 3d convolutions
#         self.conv1 = nn.Conv3d(in_channels=self.in_channels,
#                                out_channels=self.conv_channels,
#                                kernel_size=3, 
#                                padding=1,
#                                bias=True)

#         # we can also just use the torch module to make a call, but you would do that
#         # in the forward function....but we're extra today so
#         self.relu1 = nn.ReLU(inplace=True)
        
#         self.conv2 = nn.Conv3d(in_channels=self.conv_channels,
#                                out_channels=self.conv_channels,
#                                kernel_size=3,
#                                padding=1,
#                                bias=True)
        
#         self.relu2 = nn.ReLU(inplace=True)

#         # the maxpool has a stride of 2
#         self.maxpool = nn.MaxPool3d(2,2)
    
#     def forward(self, input_batch):
#         block_out = self.conv1(input_batch)
#         block_out = self.relu1(block_out)
#         block_out = self.conv2(block_out)
#         block_out = self.relu2(block_out)

#         # maxpool3d is applied to the output of this LunaBlock
#         return self.maxpool(block_out)

# class LunaModel(nn.Module):
#     def __init__(self, in_channels=1, conv_channels=8):
#         super().__init__() 

#         self.in_channels = in_channels
#         self.conv_channels = conv_channels

#         # a batch normalization layer before passed through (this is the input end)
#         self.tail_batchnorm = nn.BatchNorm3d(1) 

#         # we will utilize 4 blocks of the LunaBlock Class for our classifier
#         self.block1 = LunaBlock(in_channels=self.in_channels, conv_channels=self.conv_channels)
#         self.block2 = LunaBlock(in_channels=self.conv_channels, conv_channels=self.conv_channels * 2)
#         self.block3 = LunaBlock(in_channels=self.conv_channels * 2, conv_channels=self.conv_channels * 4)
#         self.block4 = LunaBlock(in_channels=self.conv_channels * 4, conv_channels=self.conv_channels * 8)

#         # a fully connected layer before an output is computed, softmax is applied (this is the output end)
#         # we'll out put a 1D tensor with 2 elements
#         self.head_linear = nn.Linear(1152, 2)

#         # define a softmax that can be apply to the logits 
#         self.head_softmax = nn.Softmax(dim=1)

#     def forward(self, input_batch):
#         bn_output = self.tail_batchnorm(input_batch)

#         block_out = self.block1(bn_output)
#         block_out = self.block2(block_out)
#         block_out = self.block3(block_out)
#         block_out = self.block4(block_out)

#         # the reshape the tensor to be passed thorugh the linear layer
#         # you can also use block_out.shape[0]
#         conv_flat = block_out.view(block_out.size(0), -1)
#         linear_output = self.head_linear(conv_flat)

#         # linear output is our logits, the other one is a softmax that we apply
#         # the logits, linear_outpu, will be used when we calculate the nn.CrossEntropyLoss
#         # the the probabilities which are the product of softmax will be used to classify the 
#         # samples! -- this is kind of like a one hot encoding effect
#         return linear_output, self.head_softmax(linear_output)

    
#     # here we should initialize the weights of our model -- kaming is used due to 
#     # the use of relu activation (Xavier would be used if tanh was used)
#     def _init_weights(self):
#         # self.model() is a super class attribute, and essentially what this does
#         # is loop through all the layers
#         for m in self.module():
#             # if a layer belongs to the nn.Linear or nn.Conv3d modules, their weights
#             # are initialized accordingly, via kaiming_normal initialization
#             if type(m) in {nn.Linear, nn.Conv3d}:
#                 nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
#                     bound = 1 / math.sqrt(fan_out)
#                     nn.init.normal_(m.bias, -bound, bound)

import math

from torch import nn as nn

class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)

class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)



    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)



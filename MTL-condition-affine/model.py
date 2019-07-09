import torch
import torch.nn as nn
import numpy as np

class Conditional_InstanceNorm2d(torch.nn.Module):
    """
    Conditional Instance Normalization
    introduced in https://arxiv.org/abs/1610.07629
    created and applied based on my limited understanding, could be improved
    """
    def __init__(self, task_num, in_channels):
        super(Conditional_InstanceNorm2d, self).__init__()
        self.con_IN2d = torch.nn.ModuleList([torch.nn.InstanceNorm2d(in_channels, affine=True) for i in range(task_num)])

    def forward(self, x, task_id):
        # out = torch.stack([self.con_IN2d[task_id[i]](x[i].unsqueeze(0)).squeeze_(0) for i in range(len(task_id))])
        out = self.con_IN2d[task_id](x)
        return out
    

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, affine=True, relu=True, task_num=None):
        super(ConvLayer, self).__init__()
        self.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        #self.InstanceNorm = nn.InstanceNorm2d(out_channels, affine=affine)
        self.InstanceNorm = Conditional_InstanceNorm2d(task_num, out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        #self.ReLU = nn.LeakyReLU(0.2, True)
        self.relu = relu
    
    def forward(self, x, task_id):
        if self.relu:
            out = self.ReLU(self.InstanceNorm(self.Conv2d(x), task_id))
        else:
            out = self.InstanceNorm(self.Conv2d(x), task_id)
        return out
    
    
class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=1, affine=True, relu=True, task_num=None):
        super(DeconvLayer, self).__init__()
        self.Deconv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)
        #self.InstanceNorm = nn.InstanceNorm2d(out_channels, affine=affine)
        self.InstanceNorm = Conditional_InstanceNorm2d(task_num, out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        #self.ReLU = nn.LeakyReLU(0.2, True)
        self.relu = relu
    
    def forward(self, x, task_id):
        out = self.ReLU(self.InstanceNorm(self.Deconv2d(x), task_id))
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, input_features, task_num=None):
        super(ResidualBlock, self).__init__()
        self.ReflectionPad = nn.ReflectionPad2d(1)
        self.conv1 = ConvLayer(input_features, input_features, kernel_size=3, stride=1, padding=0, task_num=task_num)
        self.conv2 = ConvLayer(input_features, input_features, kernel_size=3, stride=1, padding=0, relu=False, task_num=task_num)

    def forward(self, input_data, task_id):
        out = self.conv1(self.ReflectionPad(input_data), task_id)
        out = self.conv2(self.ReflectionPad(input_data), task_id)
        return input_data + out

    

class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=9, task_num=None):
        super(Model, self).__init__()
        self.ReflectionPad = nn.ReflectionPad2d(3)
        
        conv_layers = [ConvLayer(in_channels, 64, 7, stride=1, padding=0, task_num=task_num)]
        # Two 3 x 3 convolutional layers
        input_features = 64
        output_features = input_features * 2
        for _ in range(2):
            conv_layers.append(ConvLayer(input_features, output_features, 3, task_num=task_num))
            input_features, output_features = output_features, output_features * 2    
        self.conv_layers = nn.ModuleList(conv_layers)
        

        # Residual blocks
        res_layers = []
        for _ in range(res_blocks):
            res_layers.append(ResidualBlock(input_features, task_num=task_num))
            
        self.res_layers = nn.ModuleList(res_layers)
        
        # Two 3 x 3 deconvolutional layers
        deconv_layers = []
        output_features = input_features // 2
        for _ in range(2):
            deconv_layers.append(DeconvLayer(input_features, output_features, 3, task_num=task_num))
            input_features, output_features = output_features, output_features // 2
        self.deconv_layers = nn.ModuleList(deconv_layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_features, out_channels, 7),
            nn.Tanh()
        )

    def forward(self, input_image, task_id):
        out = self.ReflectionPad(input_image)
        for i in range(len(self.conv_layers)):
            out = self.conv_layers[i](out, task_id)
        for i in range(len(self.res_layers)):
            out = self.res_layers[i](out, task_id)
        for i in range(len(self.deconv_layers)):
            out = self.deconv_layers[i](out, task_id)
        return self.output_layer(out)
import torch
import torch.nn as nn
import numpy as np

class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        
        #linear = nn.Linear(dim_in, dim_out)
        #linear.weight.data.normal_()
        #linear.bias.data.zero_()
        
        #self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)

class fc_make_param(nn.Module):
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        param = self.transform(w).unsqueeze(2).unsqueeze(3)
        return param

class Conditional_InstanceNorm2d(torch.nn.Module):
    def __init__(self, in_channels):
        super(Conditional_InstanceNorm2d, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)

    #def forward(self, x, task_id):
    def forward(self, x, param):
        factor, bias = param.chunk(2, 1)
        result = self.norm(x)
        result = result * factor + bias  
        return result
    

class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 relu=True,
                 dim_label=None):
        super(ConvLayer, self).__init__()
        self.Conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.param = fc_make_param(dim_label, out_channels)
        self.InstanceNorm = Conditional_InstanceNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.relu = relu
    
    def forward(self, x, label):
        if self.relu:
            out = self.ReLU(self.InstanceNorm(self.Conv2d(x), self.param(label)))
        else:
            out = self.InstanceNorm(self.Conv2d(x), self.param(label))
        return out
    
    
class DeconvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 output_padding=1,
                 relu=True,
                 dim_label=None):
        super(DeconvLayer, self).__init__()
        self.Deconv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)
        self.param = fc_make_param(dim_label, out_channels)
        self.InstanceNorm = Conditional_InstanceNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        #self.ReLU = nn.LeakyReLU(0.2, True)
        self.relu = relu
    
    def forward(self, x, label):
        out = self.ReLU(self.InstanceNorm(self.Deconv2d(x), self.param(label)))
        return out
    

class ResidualBlock(nn.Module):
    def __init__(self, input_features, dim_label=None):
        super(ResidualBlock, self).__init__()
        self.ReflectionPad = nn.ReflectionPad2d(1)
        self.conv1 = ConvLayer(input_features, input_features, kernel_size=3, stride=1, padding=0, dim_label=dim_label)
        self.conv2 = ConvLayer(input_features, input_features, kernel_size=3, stride=1, padding=0, relu=False, dim_label=dim_label)

    def forward(self, input_data, label):
        out = self.conv1(self.ReflectionPad(input_data), label)
        out = self.conv2(self.ReflectionPad(input_data), label)
        return input_data + out

    

class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=9, dim_label=None):
        super(Model, self).__init__()
        self.ReflectionPad = nn.ReflectionPad2d(3)
        
        conv_layers = [ConvLayer(in_channels, 64, 7, stride=1, padding=0, dim_label=dim_label)]
        # Two 3 x 3 convolutional layers
        input_features = 64
        output_features = input_features * 2
        for _ in range(2):
            conv_layers.append(ConvLayer(input_features, output_features, 3, dim_label=dim_label))
            input_features, output_features = output_features, output_features * 2    
        self.conv_layers = nn.ModuleList(conv_layers)
        

        # Residual blocks
        res_layers = []
        for _ in range(res_blocks):
            res_layers.append(ResidualBlock(input_features, dim_label=dim_label))
            
        self.res_layers = nn.ModuleList(res_layers)
        
        # Two 3 x 3 deconvolutional layers
        deconv_layers = []
        output_features = input_features // 2
        for _ in range(2):
            deconv_layers.append(DeconvLayer(input_features, output_features, 3, dim_label=dim_label))
            input_features, output_features = output_features, output_features // 2
        self.deconv_layers = nn.ModuleList(deconv_layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_features, out_channels, 7),
            nn.Tanh()
        )

    def forward(self, input_image, label):
        out = self.ReflectionPad(input_image)
        for i in range(len(self.conv_layers)):
            out = self.conv_layers[i](out, label)
        for i in range(len(self.res_layers)):
            out = self.res_layers[i](out, label)
        for i in range(len(self.deconv_layers)):
            out = self.deconv_layers[i](out, label)
        return self.output_layer(out)
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

import torch
from torch.utils.data import Dataset
from PIL import Image


##############################################
# Residual block with two convolution layers.
##############################################

    
class Network(nn.Module):

    def __init__(self, input_shape):
        super(Network, self).__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=(3, 7, 3), padding=0)
        self.conv1_bn = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 4, kernel_size=(3, 7, 3), padding=0)
        self.conv2_bn = nn.BatchNorm3d(4)
        self.conv3 = nn.Conv3d(4, 8, kernel_size=(3, 7, 3), padding=0)
        self.conv3_bn = nn.BatchNorm3d(8)
        # self.conv4 = nn.Conv3d(8, 16, kernel_size=(3, 7, 3), padding=0)
        # self.conv4_bn = nn.BatchNorm3d(16)
        self.fc1 = nn.LazyLinear(1024)
        # self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.LazyLinear(1)

    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv1_bn(self.conv1(x)), (2, 4, 2)))
        x = F.relu(F.max_pool3d(self.conv2_bn(self.conv2(x)), (2, 4, 2)))
        x = F.relu(F.max_pool3d(self.conv3_bn(self.conv3(x)), (2, 4, 2)))
        # x = F.relu(F.max_pool3d(self.conv4_bn(self.conv4(x)), (2, 4, 2)))
        # print(x.size(0), x.size())
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return (x)
     
    
class DynamicNetwork(nn.Module):
    def __init__(self, input_shape, conv_layers, kernel_size, out_channel_ratio, FC_layers):
        super(DynamicNetwork, self).__init__()

        in_channels, height, samples, width = input_shape
       
        model = []
        
        for i in range(conv_layers):
            out_channels = in_channels * out_channel_ratio
            model += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2)
                ]
            in_channels = out_channels     

        out_size = (in_channels*height*width)/(2**(i+1))**2 #MAX POOL REDUCES H AND W BY 2 EACH TIME. THIS FACTORS OUT AS SQUARED(1/2*ITER)

        model += [
            nn.Flatten()
        ]
        
        in_channels = out_size
        FC_layer_diff = in_channels//FC_layers
        
        for j in range((FC_layers-1)):
            out_channels = in_channels - (FC_layer_diff)
            model += [
                nn.Linear(int(in_channels), int(out_channels)),
                nn.ReLU()
            ]
            # print(in_channels, out_channels)
            in_channels = out_channels   
        
        model += [
            nn.LazyLinear(1),
        ]
            
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CNN3D(nn.Module):
    def __init__(self, input_shape, num_conv=3, channel_ratio=2, kernel=3, pool_layers=2, num_fc=1, normalisation='batch'):
        super(CNN3D, self).__init__()
        self.input = input_shape
        
        in_c = self.input[0]
        
        model = []
        # nn.MaxPool3d((1, 2, 1))]
        
        out_size = in_c*self.input[1]*self.input[2]*self.input[3]
        
        for i in range(num_conv):
            conv, in_c = self._conv_layer_set(
                in_c, in_c*channel_ratio, kernel, normalisation)
            model += conv
            out_size = channel_ratio*out_size/(2**3)
            
        if pool_layers>0:
            for i in range(pool_layers-1):
                model += [nn.MaxPool3d((1, 2, 1)),
                          ]
                out_size = out_size/(2)

            if normalisation == 'batch':
                norm_layer = nn.BatchNorm3d(in_c)
            elif normalisation == 'instance':
                norm_layer = nn.InstanceNorm3d(in_c)
                
            model += [nn.MaxPool3d((1, 2, 1)),
                      nn.Conv3d(in_c, in_c, kernel_size=(1, kernel, 1), padding='same'),
                      norm_layer,
                      nn.LeakyReLU(),
                      ]
            
            out_size = out_size/(2)
                
        model += [nn.Flatten()]

        in_channels = out_size
        
         
        
        for j in range(0, (num_fc-1)):
            
            out_channels=int(out_size*(num_fc-(j+1))/num_fc)
           
            if normalisation == 'batch':
                norm_layer = nn.BatchNorm1d(int(out_channels))
            elif normalisation == 'instance':
                norm_layer = nn.InstanceNorm1d(int(out_channels))
            
            model += [
                nn.LazyLinear(out_channels),
                norm_layer,
                nn.LeakyReLU()]
            in_channels = out_channels   
        
        model += [
            nn.LazyLinear(1)
        ]    
                    
        self.model = nn.Sequential(*model)
        
            
    def _conv_layer_set(self, in_c, out_c, kernel=3, norm='batch'):
        if norm == 'batch':
            normalisation = nn.BatchNorm3d(out_c)
        elif norm == 'instance':
            normalisation = nn.InstanceNorm3d(out_c)

        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, kernel, 3), padding='same'),
        normalisation,
        nn.MaxPool3d((2, 2, 2)),
        nn.LeakyReLU(),
        )
        return conv_layer, out_c


    def forward(self, x):     
        return self.model(x)
    

class CNN3D_Designed(nn.Module):
    def __init__(self, input_shape, num_reduction=3, num_conv=3, channel_ratio=2, kernel_reduction=3, kernel=3, pool_layers=2, num_fc=1, pooling='average', normalisation='instance'):
        super(CNN3D_Designed, self).__init__()
        self.input = input_shape
        
        in_c = self.input[0]
        
        out_size = in_c*self.input[1]*self.input[2]*self.input[3]
        
        model = []
                


        for i in range(num_reduction):
            conv, in_c = self._conv_layer_set2(
                in_c, in_c*channel_ratio, kernel=kernel_reduction, 
                pooling = pooling, norm=normalisation)
            model += conv
            out_size = channel_ratio*out_size/(2)
        


        for i in range(num_conv):
            conv, in_c = self._conv_layer_set(
                in_c, in_c*channel_ratio, kernel=kernel, norm=normalisation)
            model += conv
            out_size = channel_ratio*out_size/(2**3)
            
                
        model += [nn.Flatten()]

        in_channels = out_size
        
         
        for j in range(0, (num_fc-1)):
            
            out_channels=int(out_size*(num_fc-(j+1))/num_fc)
           
            if normalisation == 'batch':
                norm_layer = nn.BatchNorm1d(int(out_channels))
            elif normalisation == 'instance':
                norm_layer = nn.InstanceNorm1d(int(out_channels))
            
            model += [
                nn.LazyLinear(out_channels),
                norm_layer,
                nn.LeakyReLU()]
            in_channels = out_channels   
        
        model += [
            nn.LazyLinear(1)
        ]    
                    
        self.model = nn.Sequential(*model)
        
            
    def _conv_layer_set(self, in_c, out_c, kernel=3, norm='batch'):
        if norm == 'batch':
            normalisation = nn.BatchNorm3d(out_c)
        elif norm == 'instance':
            normalisation = nn.InstanceNorm3d(out_c)

        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, kernel, 3), padding='same'),
        normalisation,
        nn.MaxPool3d((2, 2, 2)),
        nn.LeakyReLU(),
        )
        return conv_layer, out_c

    def _conv_layer_set2(self, in_c, out_c, kernel=3, pooling = 'average', norm='batch'):
        if norm == 'batch':
            normalisation = nn.BatchNorm3d(out_c)
        elif norm == 'instance':
            normalisation = nn.InstanceNorm3d(out_c)
            
        if pooling == 'average':
            pool_layer = nn.AvgPool3d((1, 2, 1))
        elif pooling == 'max':
            pool_layer = nn.MaxPool3d((1,2,1))

        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(1, kernel, 1), padding='same'),
        normalisation,
        pool_layer, # average pooling used initially
        nn.LeakyReLU(),
        )
        return conv_layer, out_c
    

    def forward(self, x):     
        return self.model(x)
    

class CNN3D_Designed_Constant(nn.Module):
    def __init__(self, input_shape, num_reduction=3, num_conv=3, channel_ratio=2, kernel_reduction=3, kernel=3, pool_layers=2, num_fc=1, pooling='average', normalisation='instance'):
        super(CNN3D_Designed_Constant, self).__init__()
        self.input = input_shape
        
        in_c = self.input[0]
        
        out_size = in_c*self.input[1]*self.input[2]*self.input[3]
        
        model = []
                

        for i in range(num_conv):
            conv, in_c = self._conv_layer_set(
                in_c, in_c*channel_ratio, kernel=kernel, norm=normalisation)
            model += conv
            out_size = channel_ratio*out_size/(2**4)
            
                
        model += [nn.Flatten()]

        in_channels = out_size
        
         
        for j in range(0, (num_fc-1)):
            
            out_channels=int(out_size*(num_fc-(j+1))/num_fc)
           
            if normalisation == 'batch':
                norm_layer = nn.BatchNorm1d(int(out_channels))
            elif normalisation == 'instance':
                norm_layer = nn.InstanceNorm1d(int(out_channels))
            
            model += [
                nn.LazyLinear(out_channels),
                norm_layer,
                nn.LeakyReLU()]
            in_channels = out_channels   
        
        model += [
            nn.LazyLinear(1)
        ]    
                    
        self.model = nn.Sequential(*model)
        
            
    def _conv_layer_set(self, in_c, out_c, kernel=3, norm='batch'):
        if norm == 'batch':
            normalisation = nn.BatchNorm3d(out_c)
        elif norm == 'instance':
            normalisation = nn.InstanceNorm3d(out_c)

        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, kernel, 3), padding='same'),
        normalisation,
        nn.MaxPool3d((2, 4, 2)),
        nn.LeakyReLU(),
        )
        return conv_layer, out_c
    

    def forward(self, x):     
        return self.model(x)
    

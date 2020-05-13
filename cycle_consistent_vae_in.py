import numpy as np
import torch
from torch import nn
import torch.optim as opt
from torch.nn import functional as F
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Encoder, self).__init__()

        self.conv_model = nn.Sequential(OrderedDict([
            ('conv_1',
             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True)),
            ('bn_1', nn.InstanceNorm2d(num_features=16, track_running_stats=True)),
            ('relu_1', nn.ReLU(inplace=True)),

            ('conv_2',
             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)),
            ('bn_2', nn.InstanceNorm2d(num_features=32, track_running_stats=True)),
            ('relu_2', nn.ReLU(inplace=True)),

            ('conv_3',
             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)),
            ('bn_3', nn.InstanceNorm2d(num_features=64, track_running_stats=True)),
            ('relu_3', nn.ReLU(inplace=True)),
            
            ('conv_4',
             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)),
            ('bn_4', nn.InstanceNorm2d(num_features=128, track_running_stats=True)),
            ('relu_4', nn.ReLU(inplace=True))
        ]))

        # Style embeddings (z)
        self.style_mu = nn.Linear(in_features=512, out_features=style_dim, bias=True)
        self.style_logvar = nn.Linear(in_features=512, out_features=style_dim, bias=True)

        # Class embeddings (s)
        self.class_output = nn.Linear(in_features=512, out_features=class_dim, bias=True)

    def forward(self, x):
        
        x = self.conv_model(x)
        x = x.reshape(x.size(0), x.size(1) * x.size(2) * x.size(3))
        
        style_embeddings_mu = self.style_mu(x)
        style_embeddings_logvar = self.style_logvar(x)        
        class_embeddings = self.class_output(x)
        
        return style_embeddings_mu, style_embeddings_logvar, class_embeddings
    
class Decoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Decoder, self).__init__()

        # Style embeddings input
        self.style_input = nn.Sequential(
            nn.Linear(in_features=style_dim, out_features=512, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Class embeddings input
        self.class_input = nn.Sequential(
            nn.Linear(in_features=class_dim, out_features=512, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.deconv_model = nn.Sequential(OrderedDict([
            ('deconv_1',
             nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=0, bias=True)),
            ('de_bn_1', nn.InstanceNorm2d(num_features=64, track_running_stats=True)),
            ('leakyrelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('deconv_2',
             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True)),
            ('de_bn_2', nn.InstanceNorm2d(num_features=32, track_running_stats=True)),
            ('leakyrelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('deconv_3',
             nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True)),
            ('de_bn_3', nn.InstanceNorm2d(num_features=16, track_running_stats=True)),
            ('leakyrelu_3', nn.LeakyReLU(negative_slope=0.2, inplace=True)),
            
            ('deconv_4',
             nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True))
        ]))

    def forward(self, style_embeddings, class_embeddings):
        
        style_embeddings = self.style_input(style_embeddings)
        class_embeddings = self.class_input(class_embeddings)

        x = torch.cat((style_embeddings, class_embeddings), dim=1)
        x = x.reshape(x.size(0), 256, 2, 2)
        x = self.deconv_model(x)

        return x
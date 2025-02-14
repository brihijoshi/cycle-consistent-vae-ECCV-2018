import torch
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=z_dim, out_features=2048, bias=True)),
#             ('fc_1_bn', nn.BatchNorm1d(num_features=2048)),
            ('leakyrelu_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_2', nn.Linear(in_features=2048, out_features=1024, bias=True)),
#             ('fc_2_bn', nn.BatchNorm1d(num_features=1024)),
            ('leakyrelu_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            ('fc_3', nn.Linear(in_features=1024, out_features=num_classes, bias=True))
        ]))

    def forward(self, z):
        x = self.fc_model(z)

        return x
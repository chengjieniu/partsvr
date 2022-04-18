# reference from https://github.com/charlesq34/pointnet-autoencoder/blob/master/models/model.py
# PointNet encoder, FC decoder, Using GPU chamfer's distance loss
# Data: Sep 2019
import numpy as np


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class PointD(nn.Module):
    def __init__(self, num_points = 2000):
        super(PointD, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        # self.fc3 = nn.Linear(1024, 4096)
        self.fc4 = nn.Linear(1024, self.num_points * 3)

        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points)
        return x


if __name__ == '__main__':
	sim_data = autograd.Variable(torch.randn(32,256))
	cls = PointD()
	out = cls(sim_data)
	print('class', out.size())
         


import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn.functional as F
from util.app_config import config as app_config

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
class PCDecoder_Part(nn.Module):
    def __init__(self, config):
        super(PCDecoder_Part, self).__init__()
        self.config = config
        self.pc_num_parts = config.pc_num_points_part

        self.fc1_1 = nn.Linear(256, 512)
        self.fc1_2 = nn.Linear(512, 1024)
        self.fc1_3 = nn.Linear(1024, self.pc_num_parts *3)
        self.th_1 = nn.Tanh()

        self.fc2_1 = nn.Linear(256, 512)
        self.fc2_2 = nn.Linear(512, 1024)
        self.fc2_3 = nn.Linear(1024, self.pc_num_parts *3)
        self.th_2 = nn.Tanh()

        self.fc3_1 = nn.Linear(256, 512)
        self.fc3_2 = nn.Linear(512, 1024)
        self.fc3_3 = nn.Linear(1024, self.pc_num_parts *3)
        self.th_3 = nn.Tanh()

        self.fc4_1 = nn.Linear(256, 512)
        self.fc4_2 = nn.Linear(512, 1024)
        self.fc4_3 = nn.Linear(1024, self.pc_num_parts *3)
        self.th_4 = nn.Tanh()              

        self.predict_scaling_factor = nn.Sequential(
            torch.nn.Linear(1024, 1),
        )

        self.predict_focal_length = nn.Sequential(
            torch.nn.Linear(1024, 1),
        )


    def forward(self, x, config):
        x = x.view(x.size(0), 1024)
        y = x

        # x1 = F.relu(self.fc1_1(x))
        # x1 = F.relu(self.fc1_2(x1))
        # x1 = self.th_1(self.fc1_3(x1))
        # x1 = x1.view(x1.size(0), self.pc_num_parts, 3)
        # outputs = dict()
        
        # outputs['pc_part_1'] = x1
        # outputs['points_1'] = x1
        [x1,x2,x3,x4] = x.split([256 , 256 , 256 ,256 ] , dim = -1)
        # part_1 back
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.th_1(self.fc1_3(x1))
        x1 = x1.view(x1.size(0), self.pc_num_parts, 3)

        #part_2 seat
        x2 = F.relu(self.fc2_1(x2))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.th_2(self.fc2_3(x2))
        x2 = x2.view(x2.size(0), self.pc_num_parts, 3)

        #part_3 leg
        x3 = F.relu(self.fc3_1(x3))
        x3 = F.relu(self.fc3_2(x3))
        x3 = self.th_3(self.fc3_3(x3))
        x3 = x3.view(x3.size(0), self.pc_num_parts, 3)

        #part_4 arm
        x4 = F.relu(self.fc4_1(x4))
        x4 = F.relu(self.fc4_2(x4))
        x4 = self.th_4(self.fc4_3(x4))
        x4 = x4.view(x4.size(0), self.pc_num_parts, 3)

        outputs = dict()
        outputs['pc_part_1'] = x1  #unit_cube
        outputs['pc_part_2'] = x2  #unit_cube
        outputs['pc_part_3'] = x3  #unit_cube
        outputs['pc_part_4'] = x4 #unit_cube

        if config.pc_learn_occupancy_scaling:
            scaling = self.predict_scaling_factor( y )
            outputs['scaling_factor'] = torch.sigmoid(scaling) * config.pc_occupancy_scaling_maximum
        else:
            outputs['scaling_factor'] = None
        if config.learn_focal_length:
            focal = self.predict_focal_length(y)
            outputs['focal_length'] = torch.sigmoid(focal) * config.focal_length_range  + config.focal_length_mean
        else:
            outputs['focal_length'] = None

        return outputs
        
        
    # def sample_latent(self, num_samples):
    #     return torch.randn((num_samples, 1024))

# input = torch.ones(1, 1024)
# input = Variable(input)
# config = app_config
# model = PCDecoder_Part(config)
# out=model(input,config)
# print(model)

# class PCDecoder_Part(nn.Module):
#     def __init__(self, num_points = 2048):
#         super(PCDecoder_Part, self).__init__()
#         self.num_points = num_points
#         self.fc1 = nn.Linear(100, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 512)
#         self.fc4 = nn.Linear(512, 1024)
#         self.fc5 = nn.Linear(1024, self.num_points * 3)
#         self.th = nn.Tanh()
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.th(self.fc5(x))
#         x = x.view(batchsize, 3, self.num_points)
#         return x
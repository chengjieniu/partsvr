import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
# from pc_projection import pointcloud_project, predict_focal_length, predict_scaling_factor, compute_projection

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class PCDecoder(nn.Module):
    def __init__(self):
        super(PCDecoder,self).__init__()

        self.decoder2D_pose = nn.Sequential(
            nn.Linear(24000, 1024),
            # nn.Linear(1024, 1024),
            nn.Linear(1024, 24000)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), 24000)
        x = self.decoder2D_pose(x)
        x = x.view(x.size(0), 8000, 3)
        outputs = dict()
        outputs['points_1'] = x
        return outputs

# model = PoseDecoder()
# input = torch.randn(2, 1024)
# input = Variable(input)
# out = model(input)
# print( model)



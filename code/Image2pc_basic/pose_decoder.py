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


    


class PoseDecoder(nn.Module):
    def __init__(self):
        super(PoseDecoder,self).__init__()

        self.decoder2D_pose = nn.Sequential(
            nn.Linear(1024, 4)
        )

        self.decoder2D_pose_1 = nn.Sequential(
            nn.Linear(1024, 32),
            nn.LeakyReLU()
        )

        self.decoder2D_pose_2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU()
        )
        self.decoder2D_pose_3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU()
        )
        self.decoder2D_pose_4 = nn.Sequential(
            nn.Linear(32, 4),
        )

    
    def forward(self, x, cfg):
        # x = x.view(x.size(0), 1024)
        outputs = dict()
        num_candidates = cfg.pose_predict_num_candidates

        def pose_branch(self, input, cfg):
            num_layers = cfg.pose_candidates_num_layers
            f_dim = 32
            
            t = self.decoder2D_pose_1(input)
            t = self.decoder2D_pose_2(t)
            t = self.decoder2D_pose_3(t)
            t = self.decoder2D_pose_4(t)
            return t


        if num_candidates > 1:
            outs = [pose_branch(self, x, cfg) for _ in range(num_candidates)]
            q = torch.cat(outs, dim = 1)
            q = torch.reshape(q, [-1, 4])
            if cfg.pose_predictor_student:
                outputs['pose_student'] = pose_branch(self, x, cfg)
        else:
            q = self.decoder2D_pose(x)
        
        outputs['poses'] = q
        outputs["predicted_translation"] = None
        return outputs

# model = PoseDecoder()
# input = torch.randn(2, 1024)
# input = Variable(input)
# out = model(input)
# print( model)



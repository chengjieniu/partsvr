import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence


def weights_init(m):
    classname = m.__class__.__name__
    if classname.fine('Conv')!=-1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)
class PCDecoder(nn.Module):
    def __init__(self):
        super(PCDecoder, self).__init__()
        
        self.decoder_pc_fc = nn.Sequential(
            nn.Linear(1024, 8000*3),
            # nn.LeakyReLU(),
            # nn.Linear(8000, 8000*2),
            # nn.LeakyReLU(),
            # nn.Linear(8000*2, 8000*3)
        )
        self.decoder_pc_tanh = nn.Sequential(
            nn.Tanh()
        )
        self.predict_scaling_factor = nn.Sequential(
            torch.nn.Linear(1024, 1),
        )

        self.predict_focal_length = nn.Sequential(
            torch.nn.Linear(1024, 1),
        )


    def forward(self, x, config):
        x = x.view(x.size(0), 1024)
        y = x
        x = self.decoder_pc_fc(x)
        x = x.view(x.size(0), 8000, 3)
        x = self.decoder_pc_tanh(x)
        outputs = dict()
        outputs['points_1'] = x /2.0 #unit_cube
        if config.pc_learn_occupancy_scaling:
            scaling = self.predict_scaling_factor( y)
            outputs['scaling_factor'] = torch.sigmoid(scaling) * config.pc_occupancy_scaling_maximum
        else:
            outputs['scaling_factor'] = None
        if config.learn_focal_length:
            focal = self.predict_focal_length(y)
            outputs['focal_length'] = torch.sigmoid(focal) * config.focal_length_range  + config.focal_length_mean
        else:
            outputs['focal_length'] = None

        return outputs

# input = torch.ones(1, 1024)
# input = Variable(input)

# model = PCDecoder()
# out=model(input)
# print(model)

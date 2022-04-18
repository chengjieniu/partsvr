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
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.pc_num_parts = config.pc_num_points_part
        self.pc_num_full = config.pc_num_points
        
        self.disci_part = nn.Sequential(
            nn.Linear( self.pc_num_parts*3, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512,1),
            # nn.LeakyReLU()#for wgan-gp
            nn.Sigmoid()
            # nn.Softmax()
        )
        self.disci_full = nn.Sequential(
            nn.Linear( self.pc_num_full*3, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
       
    def forward(self, x):
        x = x.view(x.size(0), -1)
        if x.shape[-1] == self.pc_num_full*3:
            
            x = self.disci_full(x)
        elif x.shape[-1] == self.pc_num_parts*3:
            x = self.disci_part(x)
        return x.squeeze(-1)
           






# input = torch.ones(1, 1024)
# input = Variable(input)

# model = PCDecoder()
# out=model(input)
# print(model)

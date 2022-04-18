import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.fine('Conv')!=-1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class Assemble_Net(nn.Module):
    def __init__(self):
        super(Assemble_Net, self).__init__()
        self.pc_fc1 = nn.Linear(8000*7 , 1024)
        self.pc_bn1 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(2048, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 24)
        self.bn3 = nn.BatchNorm1d(24)
        self.fc4 = nn.Linear(24, 24)

        self.feature_fc1 = nn.Linear(1024, 1024)
        self.feature_bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, feature):
        feature1 = feature['z_latent'].view(x.size()[0], -1)
        feature1 = F.relu(self.feature_bn1(self.feature_fc1(feature1)))
        x = x.view(x.size()[0], -1)

        x = F.relu(self.pc_bn1(self.pc_fc1(x)))
        x = torch.cat([x, feature1], -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
 
        x = self.fc4(x)
        x = x.view(x.size()[0], 4, -1)

        [scale_pre, translation_pre] = x.chunk(2, -1)
        scale = scale_pre + 1
        out = torch.cat([scale, translation_pre*0.01], -1)
        out = out.view(out.size()[0], -1)

        return out

# input = torch.ones(2, 8000, 3)
# input = Variable(input).cuda()

# model = Assemble_Net()
# out=model(input)
# print(model)
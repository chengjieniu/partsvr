# import math
import torch
from torch import nn
from torch.autograd import variable
from time import time
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, data
from tensorboardX import SummaryWriter


def _preprocess(images):
    return images * 2 - 1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("skipping initialization of ", classname)


class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        # self.args = args

        self.encoder2D_features = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.encoder2D_fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.LeakyReLU()
        )
        self.encoder2D_fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU()
        )
        self.encoder2D_fc3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU()
        )
        self.encoder2D_fc4 = nn.Sequential(
            nn.Linear(1024, 1024),
            #    nn.ReLU()
        )

        self.predict_scaling_factor = nn.Sequential(
            torch.nn.Linear(1024, 1),
        )

        self.predict_focal_length = nn.Sequential(
            torch.nn.Linear(1024, 1),
        )

    def forward(self, x, cfg):
        outputs = dict()
        x = _preprocess(x)
        x = x.view(-1, 3, 128, 128)
        x = self.encoder2D_features(x)
        x = x.view(-1, 4 * 4 * 256)
        outputs["conv_features"] = x
        x = self.encoder2D_fc1(x)
        outputs["z_latent"] = x
        x = self.encoder2D_fc2(x)
        outputs["ids"] = self.encoder2D_fc3(x)

        if cfg.pc_learn_occupancy_scaling:
            scaling = self.predict_scaling_factor(outputs["ids"])
            outputs['scaling_factor'] = torch.sigmoid(scaling) * cfg.pc_occupancy_scaling_maximum
        else:
            outputs['scaling_factor'] = None
        if cfg.learn_focal_length:
            focal = self.predict_focal_length(outputs["ids"])
            outputs['focal_length'] = torch.sigmoid(focal) * cfg.focal_length_range + cfg.focal_length_mean
        else:
            outputs['focal_length'] = None

        if cfg.predict_pose:
            outputs["poses"] = self.encoder2D_fc4(x)
        return outputs

# model = ImgEncoder()
# #model.apply(weights_init)
# inputsd = torch.rand(2,3,128,128)


# # output= model(input)
# with SummaryWriter(comment='ImgEncoder') as w:
#     w.add_graph(model, (inputsd,))
# # print(model)
# # print(output)




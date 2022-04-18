import os
import time
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
import utiltorch
from dynamicplot import DynamicPlot
from time import strftime, gmtime
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import io,data
from show_pc import ShowPC
from mydataset_gan import myDataset
from img_encoder import ImgEncoder
from pc_decoder_part import PCDecoder_Part
from discrinimator import Discriminator
from pc_projection import pointcloud_project, compute_projection
from pose_decoder import PoseDecoder
import torch.nn.functional as F
from scipy.io import savemat
from model_base import pool_single_view, preprocess
from model_pc import setup_sigma, replicate_for_multiview, tf_repeat_0, proj_loss_pose_candidates, add_student_loss

from util.app_config import config as app_config
from util.train import get_trainable_variables, get_learning_rate
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression

from PIL import Image


config = app_config
config.cuda = not config.no_cuda
if config.gpu<0 and config.cuda:
    config.gpu = 0
torch.cuda.set_device(config.gpu)
# config.saved_camera = False
if config.cuda and torch.cuda.is_available():
    print("Using CUDA on GPU ", config.gpu)
else:
    print("Not using CUDA")

# input_data_src = h5py.File('./data/image.mat')
# input_data = torch.from_numpy(input_data_src['image'][:])
print("Loading data ... ... ")

data_img2pc = myDataset(config.data_path)
def my_collate(batch):
    return batch
train_iter = torch.utils.data.DataLoader(data_img2pc,batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)

print("Loading model ... ... ")
snapshot_folder = os.path.join(config.save_path)
# encoder_img = torch.load(snapshot_folder+'//'+'encoder_img_final.pkl')
decoder_pc_part = torch.load(snapshot_folder+'//'+'decoder_pc_final.pkl')
# decoder_pose = torch.load(snapshot_folder + '//'+ 'decoder_pose_final.pkl')
# encoder_img.eval()
decoder_pc_part.eval()
# decoder_pose.eval()
print("test ... ... ")
raw_inputs = {}
outputs = {}
for batch_idx, batch in enumerate(train_iter):

            
    outputs['ids_1'] = Variable(decoder_pc_part.sample_latent(config.batch_size)).cuda()

    
     #build Decoder
    dec_pcoutputs = decoder_pc_part(outputs['ids_1'],config)
    outputs.update(dec_pcoutputs) 

    outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)


print('done')


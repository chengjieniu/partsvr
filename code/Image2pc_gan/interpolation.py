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
from show_pc_seg import ShowPC_SEG
from mydataset_gan_eval import myDataset
from img_encoder import ImgEncoder
from pc_decoder_part import PCDecoder_Part
from pc_decoder import PCDecoder
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
import show_balls

from PIL import Image

import scipy.io as scio
from scipy.io import savemat
import numpy as np

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
train_iter = torch.utils.data.DataLoader(data_img2pc,batch_size=1, shuffle=False, collate_fn=my_collate)

print("Loading model ... ... ")
snapshot_folder = os.path.join(config.save_path)
model_folder = os.path.join("testmodel//chair_pose")
encoder_img = torch.load(model_folder+'//'+'encoder_img_final.pkl', map_location='cuda:0')
decoder_pc_part = torch.load(model_folder+'//'+'decoder_pc_final.pkl', map_location='cuda:0')
# decoder_pose = torch.load('decoder_pose_final.pkl')
# for params in encoder_img.named_parameters():
#     [name, param] = params
#     if "weight" in name:
#         print(name, ':', param.data.mean())

# for params in decoder_pc_part.named_parameters():
#     [name, param] = params
#     if "weight" in name:
#         print(name, ':', param.data.mean())
print("^^^^^^^^^^^")
encoder_img.eval()
decoder_pc_part.eval()
# decoder_pose.eval()
print("test ... ... ")
raw_inputs = {}
outputs = {}
points_pred = {}
out_dir = '/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/'
synth_set = '03001627'
for batch_idx, batch in enumerate(train_iter):
    # if batch_idx == 56:
    #     print('good')
    # else:
    #     continue
    IN_Image = [x[0] for x in batch]
    IN_Mask = [x[1] for x in batch]
    IN_Cameras = [x[2] for x in batch]
    IN_Cam_pos =[x[3] for x in batch]

    IN_Image = torch.stack(IN_Image)
    IN_Mask = torch.stack(IN_Mask)
    IN_Cameras = torch.stack(IN_Cameras)
    IN_Cam_pos = torch.stack(IN_Cam_pos)
    raw_inputs['image'] = utiltorch.var_or_cuda(IN_Image)
    raw_inputs['mask'] = utiltorch.var_or_cuda(IN_Mask)
    raw_inputs['cameras'] = utiltorch.var_or_cuda(IN_Cameras)
    raw_inputs['cam_pos'] = utiltorch.var_or_cuda(IN_Cam_pos)
            
    inputs = preprocess(raw_inputs, config)


    #build Encoder
    # code = 'images' if config.predict_pose else 'images_1'
    code = 'images'
    enc_outputs = encoder_img(inputs[code],config)
    ids = enc_outputs['ids']
    outputs['conv_features'] = enc_outputs['conv_features']
    outputs['ids'] = ids
    outputs['z_latent'] = enc_outputs['z_latent']
    outputs['ids_1'] = ids

    # unsupervised case, case where convnet runs on all views, need to extract the first
    if ids.shape[0] != config.batch_size:
        ids = pool_single_view(config, ids, 0)
    outputs['ids_1'] = ids

     #build Decoder
    dec_pcoutputs = decoder_pc_part(outputs['ids_1'],config)
    outputs.update(dec_pcoutputs) 

    outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)
    # outputs['scaling_factor'] = predict_scaling_factor(config, outputs['ids_1'], is_training)
    # outputs['focal_length'] = predict_focal_length(config, outputs['ids'], is_training)
    outputs['all_rgb'] = None
    all_scaling_factors = outputs['scaling_factor']
    outputs['all_scaling_factors'] = all_scaling_factors
    #predict pose
    outputs['all_rgb'] = None
    all_scaling_factors = outputs['scaling_factor']
    outputs['all_scaling_factors'] = all_scaling_factors
    #predict pose
    if config.predict_pose:
        dec_pose = decoder_pose(enc_outputs['poses'], config)
        outputs.update(dec_pose)

    all_scaling_factors = None
    outputs['all_rgb'] = None
    all_scaling_factors = outputs['scaling_factor']
    outputs['all_scaling_factors'] = all_scaling_factors
    outputs['focal_length'] = None
    # run_projection = True
    # if run_projection:
    #     num_candidates = config.pose_predict_num_candidates
    #     all_points = replicate_for_multiview(outputs['points_1'],config)
                
    #     all_focal_length = None                                

    #     if num_candidates > 1:
    #         all_points = tf_repeat_0(all_points, num_candidates)
    #         if config.predict_translation:
    #             trans = outputs["predicted_translation"]
    #             outputs["predicted_translation"] = tf_repeat_0(trans, num_candidates)
    #         focal_length = outputs['focal_length']
    #         if focal_length is not None:
    #             all_focal_length = tf_repeat_0(focal_length, num_candidates)                        

    #     outputs['all_focal_length'] = all_focal_length
    #     outputs['all_points'] = all_points

    #     if config.pc_learn_occupancy_scaling:
    #         all_scaling_factors = replicate_for_multiview(outputs['scaling_factor'], config)
    #         if num_candidates > 1:
    #                 all_scaling_factors = tf_repeat_0(all_scaling_factors, num_candidates)
                
    #     outputs['all_scaling_factors'] = all_scaling_factors

    #     if config.pc_rgb:
    #         all_rgb = replicate_for_multiview(outputs['rgb_1'])
    #     else:
    #         all_rgb = None
    #     outputs['all_rgb'] = all_rgb


    # proj_output = compute_projection(inputs, outputs, False, config, None, len(train_iter))
    # outputs.update(proj_output)
    # # if batch_idx == 56:
    # #     print('good')
    # IN_Mask = inputs['masks'].permute(0,3,1,2)
    # IN_Mask = F.interpolate(IN_Mask,[64,64])
    # IN_Mask = IN_Mask.permute(0,2,3,1)
    # task_loss = torch.nn.MSELoss(reduction='mean')
    # total_loss = task_loss(IN_Mask,outputs['projs']) *4096  /2 
    # print('proj_loss: ', task_loss)

    # points_pred.setdefault('points', []).append(np.array(outputs['points_1'].cpu().detach().numpy()))
    # points_pred.setdefault('name',[]).append([x[4] for x in batch])

# interpolation
    if batch_idx ==533:
            inter2 = outputs["ids_1"]
            break
    if batch_idx ==195:
            inter1 = outputs["ids_1"]
dec_pcoutputs1 = decoder_pc_part(inter1,config)

plt.imshow(inputs['images_1'][0].squeeze(0).cpu().detach())
plt.savefig(snapshot_folder+'//Interpolation_video//'+str(batch_idx)+'_'+'_image.png')
plt.close()

dec_pcoutputs2 = decoder_pc_part(inter2,config)
interpola = Variable(torch.zeros( 52 ,1024)).cuda()
for i in range(50):
    x = (1-i/49.0)    
    interpola[i] = inter2 
    # interpola[i][511:768] = inter2[0, 511:768] * x +inter1[0 ,511:768] * (1-x)
    interpola[i] = inter1 * x +inter2 * (1-x)
    print('i: ', i )
    outputs = decoder_pc_part(interpola[i].unsqueeze(0).cuda(), config)
    outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)
    frame = show_balls.show_3d_point_clouds(outputs['points_1'][0].cpu().detach().numpy(), False)
    plt.imshow(frame)
    plt.axis('off')
    plt.savefig(snapshot_folder+'//Interpolation_video//'+str(batch_idx)+'__'+str(i) +'_point.png')
    plt.close()

    # for i in range(0, config.batch_size):
    # # for i in range(0, 150):
    #     plt.imshow(inputs['images_1'][i].squeeze(0).cpu().detach())
    #     plt.savefig(snapshot_folder+'//result_513_unsuper_train_woseg//'+str(batch_idx)+'_'+str(i) +'_image.png')
    #     plt.close()
    #     print('%d: %f'%(i, total_loss.item()))
    #     # ShowPC_SEG(outputs['pc_part_1'][i],outputs['pc_part_2'][i],outputs['pc_part_3'][i],outputs['pc_part_4'][i])
    #     frame = show_balls.show_3d_point_clouds(outputs['points_1'][i].cpu().detach().numpy(), True)
    #     plt.imshow(frame)
    #     plt.axis('off')
    #     plt.savefig(snapshot_folder+'//result_513_unsuper_train_woseg//'+str(batch_idx)+'_'+str(i) +'_point.png')
    #     plt.close()


# scio.savemat(out_dir + synth_set + 'pred_test_gan', {'name': points_pred['name'], 'points': points_pred['points']})  
print('done')


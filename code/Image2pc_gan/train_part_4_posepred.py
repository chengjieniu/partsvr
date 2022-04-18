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
from mydataset_gan import myDataset
from img_encoder import ImgEncoder
from pc_decoder_part import PCDecoder_Part

from pointnet1 import PointNetCls
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
from tensorboardX import SummaryWriter
from sklearn.neighbors import NearestNeighbors
import random
import numpy as np

import warnings

warnings.filterwarnings('ignore')
writer = SummaryWriter()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train():

    # manualSeed = random.randint(1, 10000) # fix seed
    # print("Random Seed: ", manualSeed)
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    config = app_config
    config.cuda = not config.no_cuda
    if config.gpu<0 and config.cuda:
        config.gpu = 0
    torch.cuda.set_device(config.gpu)
    if config.cuda and torch.cuda.is_available():
        print("Using CUDA on GPU ", config.gpu)
    else:
        print("Not using CUDA")

    print("Get model ... ... ")
    
    encoder_img = ImgEncoder()
    decoder_pc_part = PCDecoder_Part(config)
    discri_part_1 = PointNetCls(k = 2, num_points = 2000)
    discri_part_2 = PointNetCls(k = 2, num_points = 2000)
    discri_part_3 = PointNetCls(k = 2, num_points = 2000)
    discri_part_4 = PointNetCls(k = 2, num_points = 2000)
    discri_full = PointNetCls(k = 2, num_points = 8000)

    print(decoder_pc_part)
    print(discri_full)

    encoder_img.apply(weights_init)
    decoder_pc_part.apply(weights_init)

    discri_part_1.apply(weights_init)
    discri_part_2.apply(weights_init)
    discri_part_3.apply(weights_init)
    discri_part_4.apply(weights_init)
    discri_full.apply(weights_init)

    if config.predict_pose:
        decoder_pose = PoseDecoder()

    if config.cuda:
        encoder_img.cuda()
        decoder_pc_part.cuda()
        if config.predict_pose:
            decoder_pose.cuda()
        discri_part_1.cuda()
        discri_part_2.cuda()
        discri_part_3.cuda()
        discri_part_4.cuda()
        discri_full.cuda()

    encoder_img_opt = torch.optim.Adagrad(encoder_img.parameters(), lr = 1e-4)
    decoder_pc_part_opt = torch.optim.Adagrad(decoder_pc_part.parameters(), lr = 1e-4)
    if config.predict_pose:
        decoder_pose_opt = torch.optim.Adagrad(decoder_pose.parameters(), lr = 1e-4, weight_decay = 1e-6) 
    discri_part_1_opt = torch.optim.Adagrad(discri_part_1.parameters(), lr = 1e-4)
    discri_part_2_opt = torch.optim.Adagrad(discri_part_2.parameters(), lr = 1e-4)
    discri_part_3_opt = torch.optim.Adagrad(discri_part_3.parameters(), lr = 1e-4)
    discri_part_4_opt = torch.optim.Adagrad(discri_part_4.parameters(), lr = 1e-4)
    discri_full_opt = torch.optim.Adagrad(discri_full.parameters(), lr = 1e-4)

    print("Loading data ... ... ", end = '', flush = True)

    data_img2pc = myDataset(config.data_path)
    def my_collate(batch):
        return batch
    train_iter = torch.utils.data.DataLoader(data_img2pc, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
  

    print("Start training ... ...")

    start = time.time()


    if config.save_snapshot:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        snapshot_folder = os.path.join(config.save_path, 'snapshots_'+ strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        if not os.path.exists(snapshot_folder):
            os.makedirs(snapshot_folder)
   
    total_iter = config.epochs * len(train_iter)

    
    is_training = True
    outputs = {}
    criterion = torch.nn.NLLLoss( reduction = 'mean' )
    raw_inputs = {}
    params = {}
    params['_gauss_sigma'] = None
    params['_gauss_kernel'] = None
    params['_sigma_rel'] = None
    params['_alignment_to_canonical'] =None
    mse_loss = torch.nn.MSELoss(reduction = 'mean' )
    train_discri = True
    n_iter = 0
    type_back = (torch.zeros(config.batch_size, 2000, 4).index_fill(2, torch.tensor([3]), 1).float()).cuda()
    type_seat = (torch.zeros(config.batch_size, 2000, 4).index_fill(2, torch.tensor([2]), 1).float()).cuda()
    type_leg = (torch.zeros(config.batch_size, 2000, 4).index_fill(2, torch.tensor([1]), 1).float()).cuda()
    type_arm = (torch.zeros(config.batch_size, 2000, 4).index_fill(2, torch.tensor([0]), 1).float()).cuda()
    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(train_iter):
            n_iter = n_iter+1
            #prepare the parameters such as kernal and sigma
            params['_global_step'] = epoch*len(train_iter) + batch_idx 
            params['_sigma_rel'], params['_gauss_sigma'], params['_gauss_kernel'] = setup_sigma(config, params['_global_step'], len(train_iter))
          
            
            IN_Image = [x[0] for x in batch]
            IN_Mask = [x[1] for x in batch]
            IN_Cameras = [x[2] for x in batch]
            IN_Cam_pos =[x[3] for x in batch]

            IN_PC_Part1 = [x[4] for x in batch]
            IN_PC_Part2 = [x[5] for x in batch]
            IN_PC_Part3 = [x[6] for x in batch]
            IN_PC_Part4 = [x[7] for x in batch]
            IN_PC_Full = [x[8] for x in batch]


            IN_Image = torch.stack(IN_Image)
            IN_Mask = torch.stack(IN_Mask)
            IN_Cameras = torch.stack(IN_Cameras)
            IN_Cam_pos = torch.stack(IN_Cam_pos)
            IN_PC_Part1 = torch.stack(IN_PC_Part1)
            IN_PC_Part2 = torch.stack(IN_PC_Part2)
            IN_PC_Part3 = torch.stack(IN_PC_Part3)
            IN_PC_Part4 = torch.stack(IN_PC_Part4)
            IN_PC_Full = torch.stack(IN_PC_Full)  

            IN_Image = utiltorch.var_or_cuda(IN_Image)
            IN_Mask = utiltorch.var_or_cuda(IN_Mask)
            IN_Cameras = utiltorch.var_or_cuda(IN_Cameras)
            IN_Cam_pos = utiltorch.var_or_cuda(IN_Cam_pos)
            
            IN_PC_Part1 = utiltorch.var_or_cuda(IN_PC_Part1)
            IN_PC_Part2 = utiltorch.var_or_cuda(IN_PC_Part2)
            IN_PC_Part3 = utiltorch.var_or_cuda(IN_PC_Part3)
            IN_PC_Part4 = utiltorch.var_or_cuda(IN_PC_Part4)
            IN_PC_Full = utiltorch.var_or_cuda(IN_PC_Full)

            raw_inputs['image'] = utiltorch.var_or_cuda(IN_Image)
            raw_inputs['mask'] = utiltorch.var_or_cuda(IN_Mask)
            raw_inputs['cameras'] = utiltorch.var_or_cuda(IN_Cameras)
            raw_inputs['cam_pos'] = utiltorch.var_or_cuda(IN_Cam_pos)

            inputs = preprocess(raw_inputs, config)        

            discri_part_1_opt.zero_grad()
            discri_part_2_opt.zero_grad()
            discri_part_3_opt.zero_grad()
            discri_part_4_opt.zero_grad()
            discri_full_opt.zero_grad()

            #build Encoder
            code = 'images' if config.predict_pose else 'images_1'
            enc_outputs = encoder_img(inputs[code], config)
            ids = enc_outputs['ids']
            outputs['conv_features'] = enc_outputs['conv_features']
            outputs['ids'] = ids
            outputs['z_latent'] = enc_outputs['z_latent']

            # unsupervised case, case where convnet runs on all views, need to extract the first
            if ids.shape[0] != config.batch_size:
                ids = pool_single_view(config, ids, 0)
            outputs['ids_1'] = ids
            # outputs['ids_1'] = Variable(torch.randn(IN_PC_Full.shape[0], 1024)).cuda()
            # outputs['ids_1'] = Variable(torch.cat(torch.randn(IN_PC_Full.shape[0], 256), torch.randn(IN_PC_Full.shape[0], 256), torch.randn(IN_PC_Full.shape[0], 256),torch.randn(IN_PC_Full.shape[0], 256), 0)).cuda()
            #build Decoder
            dec_pcoutputs = decoder_pc_part(outputs['ids_1'], config)
            outputs.update(dec_pcoutputs)


            #part gan      
            real_label_part = Variable(torch.ones(IN_PC_Part2.shape[0], dtype = torch.long)).cuda()  
            fake_label_part = Variable(torch.zeros(IN_PC_Part2.shape[0], dtype = torch.long)).cuda() 

            if train_discri:

                # #part 1 back
                real_out1, a = discri_part_1(IN_PC_Part1)
                loss_real_part1 = F.nll_loss(real_out1.squeeze(),real_label_part)
                
                fake_out1, a  = discri_part_1(outputs['pc_part_1'].detach() )
                loss_fake_part1 = F.nll_loss(fake_out1.squeeze(), fake_label_part)     

                d1_loss = (loss_fake_part1+loss_real_part1)/2
            
            

                #part 2 
                real_out2, a  = discri_part_2(IN_PC_Part2)
                loss_real_part2 = F.nll_loss(real_out2.squeeze(0),real_label_part)
                
                fake_out2, a  = discri_part_2(outputs['pc_part_2'].detach() )
                loss_fake_part2 = F.nll_loss(fake_out2.squeeze(0), fake_label_part)     

                d2_loss = (loss_fake_part2+loss_real_part2)/2
                
                
                

                #part 3 leg
                real_out3 , a = discri_part_3(IN_PC_Part3)
                loss_real_part3 = F.nll_loss(real_out3.squeeze(0),real_label_part)
                
                fake_out3 , a = discri_part_3(outputs['pc_part_3'].detach())
                loss_fake_part3 = F.nll_loss(fake_out3.squeeze(0), fake_label_part)     

                d3_loss = (loss_fake_part3+loss_real_part3)/2
                


                #part 4 arm
                real_out4 , a = discri_part_4(IN_PC_Part4)
                loss_real_part4 = F.nll_loss(real_out4.squeeze(0),real_label_part)
                
                fake_out4, a  = discri_part_4(outputs['pc_part_4'].detach() )
                loss_fake_part4 = F.nll_loss(fake_out4.squeeze(0), fake_label_part)     

                d4_loss = (loss_fake_part4+loss_real_part4 )/2                          
        
                # connect to full shape
                outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)
               
                # #global 
                real_label_full = Variable(torch.ones(IN_PC_Full.shape[0], dtype = torch.long)).cuda()  
                fake_label_full = Variable(torch.zeros(IN_PC_Full.shape[0], dtype = torch.long)).cuda()  

                real_out_full, a  = discri_full(IN_PC_Full)
                loss_real_full= F.nll_loss(real_out_full,real_label_full)
                    
                fake_out_full, a = discri_full(outputs['points_1'])
                loss_fake_full = F.nll_loss(fake_out_full, fake_label_full)     

                dfull_loss = (loss_fake_full + loss_real_full)/2
            

                # dis_loss =  (d1_loss + d2_loss + d3_loss + d4_loss )/(config.batch_size) *8 *5
                dis_loss  = (d1_loss  + d2_loss + d3_loss + d4_loss + dfull_loss)/5
                # dis_loss  = (d1_loss  + d2_loss + d3_loss + d4_loss)/4
                # dis_loss = d1_loss
                dis_loss.backward(retain_graph=True)


                discri_part_1_opt.step()
                discri_part_2_opt.step()
                discri_part_3_opt.step()
                discri_part_4_opt.step()                
                discri_full_opt.step()

            encoder_img_opt.zero_grad()
            decoder_pc_part_opt.zero_grad()
            if config.predict_pose:
                decoder_pose_opt.zero_grad()
            # #generate_loss
            fake_out_g1, a = discri_part_1(outputs['pc_part_1'])
            fake_out_g2, a = discri_part_2(outputs['pc_part_2'] )
            fake_out_g3, a = discri_part_3(outputs['pc_part_3'] )
            fake_out_g4, a = discri_part_4(outputs['pc_part_4'] )
            fake_out_gfull, a = discri_full(outputs['points_1'] )
            g1_loss = F.nll_loss(fake_out_g1, real_label_part)  
            g2_loss = F.nll_loss(fake_out_g2, real_label_part)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            g3_loss = F.nll_loss(fake_out_g3, real_label_part)
            g4_loss = F.nll_loss(fake_out_g4, real_label_part)
            gfull_loss = F.nll_loss(fake_out_gfull, real_label_full)

            #predict pose
            if config.predict_pose:
                dec_pose = decoder_pose(enc_outputs['poses'], config)
                outputs.update(dec_pose)

            # all_scaling_factors = None
            # outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)
            outputs['all_rgb'] = None
            all_scaling_factors = outputs['scaling_factor']
            outputs['all_scaling_factors'] = all_scaling_factors
            outputs['focal_length'] = None
            run_projection = True
            if run_projection:
                num_candidates = config.pose_predict_num_candidates
                all_points = replicate_for_multiview(outputs['points_1'],config)
                
                all_focal_length = None

                if num_candidates > 1:
                    all_points = tf_repeat_0(all_points, num_candidates)
                    if config.predict_translation:
                        trans = outputs["predicted_translation"]
                        outputs["predicted_translation"] = tf_repeat_0(trans, num_candidates)
                    focal_length = outputs['focal_length']
                    if focal_length is not None:
                        all_focal_length = tf_repeat_0(focal_length, num_candidates)                        

                outputs['all_focal_length'] = all_focal_length
                outputs['all_points'] = all_points

                if config.pc_learn_occupancy_scaling:
                    all_scaling_factors = replicate_for_multiview(outputs['scaling_factor'], config)
                    if num_candidates > 1:
                        all_scaling_factors = tf_repeat_0(all_scaling_factors, num_candidates)
                
                outputs['all_scaling_factors'] = all_scaling_factors

                if config.pc_rgb:
                    all_rgb = replicate_for_multiview(outputs['rgb_1'])
                else:
                    all_rgb = None
                outputs['all_rgb'] = all_rgb

            proj_output = compute_projection(inputs, outputs, is_training, config, params, len(train_iter))
            outputs.update(proj_output)


            # Vox = torch.randn(1,64,64,1)
            # #pc1 = torch.randn(1,8000,3)
            # pose1 = torch.zeros(4,4)

            #loss
            IN_Mask = inputs['masks'].permute(0,3,1,2)
            IN_Mask = F.interpolate(IN_Mask,[64,64])
            IN_Mask = IN_Mask.permute(0,2,3,1)
            # task_loss = torch.nn.MSELoss(reduction='mean').cuda()
            if num_candidates >1 :
                gt_tensor, pred_tensor, min_loss = proj_loss_pose_candidates(IN_Mask, outputs['projs'], inputs, config)
                proj_loss = task_loss(gt_tensor, pred_tensor)/(config.batch_size)
                if config.pose_predictor_student:
                    student_loss = add_student_loss(inputs, outputs, min_loss, config)/(config.batch_size) * config.pose_predictor_student_loss_weight
                    total_loss = student_loss + proj_loss
            else:
                total_loss = mse_loss(IN_Mask,outputs['projs']) * 4096

            #purity loss
            outputs['pc_part_1_label'] = torch.cat([outputs['pc_part_1'], type_back], 2)
            outputs['pc_part_2_label'] = torch.cat([outputs['pc_part_2'], type_seat], 2)
            outputs['pc_part_3_label'] = torch.cat([outputs['pc_part_3'], type_leg], 2)
            outputs['pc_part_4_label'] = torch.cat([outputs['pc_part_4'], type_arm], 2)

            outputs['points_1_label'] =  torch.cat((outputs['pc_part_1_label'],outputs['pc_part_2_label'],outputs['pc_part_3_label'],outputs['pc_part_4_label']),1)
            loss_purity = 0
            for i in range(config.batch_size):
                neigh = NearestNeighbors(n_neighbors=5, radius = 0.1)
                neigh.fit(outputs['points_1_label'][i,:, 0:3].clone().cpu().detach().numpy())
                D,I = neigh.kneighbors(outputs['points_1_label'][i,:, 0:3].clone().cpu().detach().numpy())
                loss_purity = loss_purity + mse_loss(outputs['points_1_label'][i, I[:, 1], 3:7],outputs['points_1_label'][i, I[:, 0], 3:7])\
                    + mse_loss(outputs['points_1_label'][i, I[:, 2], 3:7],outputs['points_1_label'][i, I[:, 0], 3:7])\
                        + mse_loss(outputs['points_1_label'][i, I[:, 3], 3:7],outputs['points_1_label'][i, I[:, 0], 3:7])\
                            + mse_loss(outputs['points_1_label'][i, I[:, 4], 3:7],outputs['points_1_label'][i, I[:, 0], 3:7])


            gen_loss = (g1_loss +g2_loss +g3_loss +g4_loss + gfull_loss ) /5 
            # gen_loss = (g1_loss +g2_loss +g3_loss +g4_loss)/4
            # gen_loss = g1_loss
            loss = total_loss  * 0.05 + gen_loss + loss_purity * 0.0667
            # loss = gen_loss
            loss.backward(retain_graph=True)
            if config.predict_pose:
                decoder_pose_opt.step()
            decoder_pc_part_opt.step()
            encoder_img_opt.step()  

            writer.add_scalar('Loss/loss', loss.item(), n_iter)
            writer.add_scalar('Loss/recon_loss', total_loss.item(), n_iter)
            writer.add_scalar('Loss/dis_loss', dis_loss.item(), n_iter)
            writer.add_scalar('Loss/gen_loss', gen_loss.item(), n_iter) 
            writer.add_scalar('Loss/purity_loss', loss_purity.item(), n_iter)     
            
            print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)), 'epock: %d, iteration: %d/%d, total_loss: %f, dis_loss: %f, gen_loss: %f' %(epoch+1, 1+batch_idx, len(train_iter), total_loss.item(), dis_loss, gen_loss))

        # if epoch%10 == 0:
 
        #     plt.savefig(str(epoch)+'.jpg')
        #     torch.save(encoder_img, 'encoder_img.pkl')
        #     torch.save(decoder_pc_part, 'decoder_pc_part.pkl')
        #     torch.save(discri_part_1, 'discri_part_1.pkl')
        #     torch.save(discri_part_2, 'discri_part_2.pkl')
        #     torch.save(discri_part_3, 'discri_part_3.pkl')
        #     torch.save(discri_part_4, 'discri_part_4.pkl')
        #     torch.save(discri_full, 'discri_full.pkl')


        if epoch%100==0:
        
            torch.save(encoder_img, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_encoder'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            torch.save(decoder_pc_part, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_decoderpc_part'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            if config.predict_pose:
                torch.save(decoder_pose, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_posedec'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            # torch.save(discri_part_1, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_1'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_2, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_2'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_3, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_3'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_4, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_4'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_full, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_full'+str(round(total_loss.item(),2))+'.pkl')
       
            
    print('Finished Traning')   
    torch.save(encoder_img, snapshot_folder+'//'+'encoder_img_final.pkl')
    torch.save(decoder_pc_part, snapshot_folder+'//'+'decoder_pc_final.pkl')
    if config.predict_pose:
        torch.save(decoder_pose, snapshot_folder+'//'+'decoder_pose_final.pkl')       
    # torch.save(discri_part_1, snapshot_folder+'//'+'_discri_part_1'+'_final.pkl')
    # torch.save(discri_part_2, snapshot_folder+'//'+'_discri_part_2'+'_final.pkl')
    # torch.save(discri_part_3, snapshot_folder+'//'+'_discri_part_3'+'_final.pkl')
    # torch.save(discri_part_4, snapshot_folder+'//'+'_discri_part_4'+'_final.pkl')
                     


        
if __name__ == "__main__":
    train()


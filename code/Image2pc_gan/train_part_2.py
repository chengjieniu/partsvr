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
# from pc_decoder_part import PCDecoder_Part
# from pc_decoder import PCDecoder
# from discrinimator import Discriminator
from pointnetD import PointD
from pointnet1 import PointNetCls
from pc_projection import pointcloud_project, compute_projection
from pose_decoder import PoseDecoder
import torch.nn.functional as F
from scipy.io import savemat
from model_base import pool_single_view, preprocess
from model_pc import setup_sigma, replicate_for_multiview, tf_repeat_0, proj_loss_pose_candidates, add_student_loss
from assembly_net import Assemble_Net

from util.app_config import config as app_config
from util.train import get_trainable_variables, get_learning_rate
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression
from tensorboardX import SummaryWriter

from PIL import Image
writer = SummaryWriter()
def adjust_learning_rate(optimizer, epoch, config, t=10):
    """Sets the learning rate to the initial LR decayed by 10 every t epochs，default=10"""
    new_lr = config.learning_rate * (0.1 ** (epoch // t))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train():
    # p1 = torch.randn(1,4)
    # p2 = torch.randn(1,8000,3)

    config = app_config
    config.cuda = not config.no_cuda
    #if config.gpu<0 and config.cuda:
    config.gpu = 3
    torch.cuda.set_device(config.gpu)
    if config.cuda and torch.cuda.is_available():
        print("Using CUDA on GPU ", config.gpu)
    else:
        print("Not using CUDA")

    print("Get model ... ... ")
    
    # encoder_img = ImgEncoder()
    # deback = PointD()
    # deseat = PointD()
    # delegs = PointD()
    # dearm = PointD()

    snapshot_folder = os.path.join(config.save_path)

    encoder_img = torch.load(snapshot_folder+'//'+'encoder_test.pkl')
    deback = torch.load(snapshot_folder+'//'+'definal_back.pkl')
    deseat = torch.load(snapshot_folder+'//'+'definal_seat.pkl')
    delegs = torch.load(snapshot_folder+'//'+'definal_leg.pkl')
    dearm = torch.load(snapshot_folder+'//'+'definal_arm.pkl')
    # assembel = torch.load(snapshot_folder+'//'+'assembly.pkl')
    # encoder_img.train()
    deback.train()
    deseat.train()
    delegs.train()
    dearm.train()
    # assembel.train()

    # discri_full = Discriminator(config)
    discri_part_1 = PointNetCls()
    discri_part_2 = PointNetCls()
    discri_part_3 = PointNetCls()
    discri_part_4 = PointNetCls()
    # discri_full = PointNetCls()

    if config.predict_pose:
        decoder_pose = PoseDecoder()

    encoder_img_opt = torch.optim.Adam(encoder_img.parameters(), lr = 1e-4, weight_decay = 1e-6)
    deback_opt = torch.optim.Adam(deback.parameters(), lr = 1e-4, weight_decay = 1e-6)
    deseat_opt = torch.optim.Adam(deseat.parameters(), lr = 1e-4, weight_decay = 1e-6)
    delegs_opt = torch.optim.Adam(delegs.parameters(), lr = 1e-4, weight_decay = 1e-6)
    dearm_opt = torch.optim.Adam(dearm.parameters(), lr = 1e-4, weight_decay = 1e-6)
    # assembel_opt = torch.optim.Adam(assembel.parameters(), lr = 1e-4, weight_decay = 1e-6)
    if config.predict_pose:
        decoder_pose_opt = torch.optim.Adam(decoder_pose.parameters(), lr = 1e-4,weight_decay = 1e-6) 
    discri_part_1_opt = torch.optim.Adam(discri_part_1.parameters(), lr = 1e-6, weight_decay = 1e-6)
    discri_part_2_opt = torch.optim.Adam(discri_part_2.parameters(), lr = 1e-6, weight_decay = 1e-6)
    discri_part_3_opt = torch.optim.Adam(discri_part_3.parameters(), lr = 1e-6, weight_decay = 1e-6)
    discri_part_4_opt = torch.optim.Adam(discri_part_4.parameters(), lr = 1e-6, weight_decay = 1e-6)
    # discri_full_opt = torch.optim.Adam(discri_full.parameters(), lr = 1e-4, weight_decay = 1e-8)

    # train_dir = config.checkpoint_dir
    # if not os.path.exists(train_dir):
    #     os.mknod(train_dir)     
    # else:

    #     encoder_img= torch.load('encoder_img.pkl')
    #     decoder_pc_part = torch.load('decoder_pc_part.pkl')
    #     discri_part_1 = torch.load('discri_part_1.pkl')
    #     discri_part_2 = torch.load('discri_part_2.pkl')
    #     discri_part_3 = torch.load('discri_part_3.pkl')
    #     discri_part_4 = torch.load('discri_part_4.pkl')
    #     # discri_full = torch.load('discri_full.pkl')
    #     encoder_img.train()
    #     decoder_pc_part.train()
    #     discri_part_1.train()
    #     discri_part_2.train()
    #     discri_part_3.train()
    #     discri_part_4.train()
        # discri_full.train()
  
    if config.cuda:
        encoder_img.cuda()
        deback.cuda()
        deseat.cuda()
        delegs.cuda()
        dearm.cuda()
        if config.predict_pose:
            decoder_pose.cuda()
        discri_part_1.cuda()
        discri_part_2.cuda()
        discri_part_3.cuda()
        discri_part_4.cuda()
        # discri_full.cuda()

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
    criterion = torch.nn.BCELoss(reduction='mean')
    raw_inputs = {}
    params = {}
    params['_gauss_sigma'] = None
    params['_gauss_kernel'] = None
    params['_sigma_rel'] = None
    params['_alignment_to_canonical'] =None

    train_discri = True

            
    type_back = (torch.zeros(config.batch_size, 2000, 4).index_fill(2, torch.tensor([3]), 1).float()).cuda()
    type_seat = (torch.zeros(config.batch_size, 2000, 4).index_fill(2, torch.tensor([2]), 1).float()).cuda()
    type_leg = (torch.zeros(config.batch_size, 2000, 4).index_fill(2, torch.tensor([1]), 1).float()).cuda()
    type_arm = (torch.zeros(config.batch_size, 2000, 4).index_fill(2, torch.tensor([0]), 1).float()).cuda()

    n_iter = 0
    for epoch in range(0, config.epochs):
        for batch_idx, batch in enumerate(train_iter):
            n_iter = n_iter + 1
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

            # IN_Image = train_iter.dataset[0].unsqueeze(0)
            # IN_Mask = train_iter.dataset[1].unsqueeze(0)
            # IN_PC_Part1 = train_iter.dataset[2].unsqueeze(0)
            # IN_PC_Part2 = train_iter.dataset[3].unsqueeze(0)
            # IN_PC_Part3 = train_iter.dataset[4].unsqueeze(0)
            # IN_PC_Part4 = train_iter.dataset[5].unsqueeze(0)
            # IN_PC_Full = train_iter.dataset[6].unsqueeze(0)

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

            encoder_img_opt.zero_grad()
            deback_opt.zero_grad()
            deseat_opt.zero_grad()
            delegs_opt.zero_grad()
            dearm_opt.zero_grad()
            # assembel_opt.zero_grad()

            if config.predict_pose:
                decoder_pose_opt.zero_grad()

            discri_part_1_opt.zero_grad()
            discri_part_2_opt.zero_grad()
            discri_part_3_opt.zero_grad()
            discri_part_4_opt.zero_grad()
            # discri_full_opt.zero_grad()

            #build Encoder
            code = 'images' if config.predict_pose else 'images_1'
            enc_outputs = encoder_img(inputs[code], config)
            ids = enc_outputs['ids']
            outputs['conv_features'] = enc_outputs['conv_features']
            outputs['ids'] = ids
            outputs['z_latent'] = enc_outputs['z_latent']
            outputs['scaling_factor'] = enc_outputs['scaling_factor']
            outputs['focal_length'] = enc_outputs['focal_length']

            # unsupervised case, case where convnet runs on all views, need to extract the first
            if ids.shape[0] != config.batch_size:
                ids = pool_single_view(config, ids, 0)
            outputs['ids_1'] = ids

            #build Decoder
            if ids.shape[0] != config.batch_size:
                ids = pool_single_view(config, ids, 0)
            outputs['ids_1'] = ids
            fe_back, fe_seat, fe_legs, fe_arm = torch.chunk(outputs['ids_1'], 4, dim = -1)
 
            outputs['pc_part_1'] = deback(fe_back) 
            outputs['pc_part_2'] = deseat(fe_seat)
            outputs['pc_part_3'] = delegs(fe_legs)
            outputs['pc_part_4'] = dearm(fe_arm)

            # #part gan      
            real_label_part = Variable(torch.ones(IN_PC_Part2.shape[0])).cuda()  
            fake_label_part = Variable(torch.zeros(IN_PC_Part2.shape[0])).cuda() 

            if train_discri:

                #part gan      
                # real_label_part = Variable(torch.ones(IN_PC_Part2.shape[0])).cuda()  
                # fake_label_part = Variable(torch.zeros(IN_PC_Part2.shape[0])).cuda()

                #part 1 back
                real_out1 = discri_part_1(IN_PC_Part1)
                loss_real_part1 = criterion(real_out1,real_label_part)
                
                fake_out1 = discri_part_1(outputs['pc_part_1'].detach() * 2)
                loss_fake_part1 = criterion(fake_out1, fake_label_part)     

                d1_loss = loss_fake_part1 + loss_real_part1

                # for i, (name3, param3) in enumerate(discri_part_1.named_parameters()):
                #     if 'bn' not in name3:
                #         writer.add_histogram('para/discri_part_1'+name3, param3, 0)
            
            

                #part 2 
                real_out2 = discri_part_2(IN_PC_Part2)
                loss_real_part2 = criterion(real_out2,real_label_part)
                
                fake_out2 = discri_part_2(outputs['pc_part_2'].detach() * 2)
                loss_fake_part2 = criterion(fake_out2, fake_label_part)     

                d2_loss = loss_fake_part2 + loss_real_part2     


                #part 3 leg
                real_out3 = discri_part_3(IN_PC_Part3)
                loss_real_part3 = criterion(real_out3,real_label_part)
                
                fake_out3 = discri_part_3(outputs['pc_part_3'].detach() * 2)
                loss_fake_part3 = criterion(fake_out3, fake_label_part)     

                d3_loss = loss_fake_part3 + loss_real_part3


                #part 4 arm
                real_out4  = discri_part_4(IN_PC_Part4)
                loss_real_part4 = criterion(real_out4,real_label_part)
                
                fake_out4 = discri_part_4(outputs['pc_part_4'].detach() * 2)
                loss_fake_part4 = criterion(fake_out4, fake_label_part)     

                d4_loss = loss_fake_part4 + loss_real_part4            
       
                # # #connect to full shape
                # outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)
               
                # # # #global 
                # real_label_full = Variable(torch.ones(IN_PC_Full.shape[0])).cuda()  
                # fake_label_full = Variable(torch.zeros(IN_PC_Full.shape[0])).cuda()  

                # real_out_full  = discri_full(IN_PC_Full)
                # loss_real_full= criterion(real_out_full,real_label_full)
                
                # fake_out_full = discri_full(outputs['points_1'].detach() * 2)
                # loss_fake_full = criterion(fake_out_full, fake_label_full)     

                # dfull_loss = loss_fake_full + loss_real_full
            

                dis_loss =  d1_loss + d2_loss + d3_loss + d4_loss 
                # dis_loss  = (d1_loss  + d2_loss + d3_loss + d4_loss + dfull_loss)/(config.batch_size) *5 *8
                dis_loss.backward(retain_graph=True)

                discri_part_1_opt.step()
                discri_part_2_opt.step()
                discri_part_3_opt.step()
                discri_part_4_opt.step()
                
                # discri_full_opt.step()
                dis_accuracy = (torch.sum( real_out1 + real_out2 + real_out3 + real_out4 )/ (config.batch_size * 4) + \
                                (1 - torch.sum(fake_out1 +fake_out2 +fake_out3 +fake_out4) / (config.batch_size * 4))) / 2
                print('dis_accuracy: ', dis_accuracy.item())
                writer.add_scalar('Accuracy/dis_accuracy', dis_accuracy, n_iter)

            # #generate_loss
            fake_out_g1 = discri_part_1(outputs['pc_part_1'] * 2)
            fake_out_g2 = discri_part_2(outputs['pc_part_2'] * 2)
            fake_out_g3 = discri_part_3(outputs['pc_part_3'] * 2)
            fake_out_g4 = discri_part_4(outputs['pc_part_4'] * 2)
            # fake_out_gfull = discri_full(outputs['points_1'] * 2)
            g1_loss = criterion(fake_out_g1, real_label_part)  
            g2_loss = criterion(fake_out_g2, real_label_part)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            g3_loss = criterion(fake_out_g3, real_label_part)
            g4_loss = criterion(fake_out_g4, real_label_part)
            # gfull_loss = criterion(fake_out_gfull, real_label_full)
            gen_accuracy = torch.mean(fake_out_g1 + fake_out_g2 + fake_out_g3 + fake_out_g4 )/4
            print('gen_accuracy : ' , gen_accuracy.item())
            writer.add_scalar('Accuracy/gen_accuracy', gen_accuracy, n_iter)


            # outputs['pc_part_1_pre'] = outputs['pc_part_1']
            # outputs['pc_part_2_pre'] = outputs['pc_part_2']
            # outputs['pc_part_3_pre'] = outputs['pc_part_3']
            # outputs['pc_part_4_pre'] = outputs['pc_part_4']

            # outputs['pc_part_1'] = torch.cat([outputs['pc_part_1'], type_back], 2)
            # outputs['pc_part_2'] = torch.cat([outputs['pc_part_2'], type_seat], 2)
            # outputs['pc_part_3'] = torch.cat([outputs['pc_part_3'], type_leg], 2)
            # outputs['pc_part_4'] = torch.cat([outputs['pc_part_4'], type_arm], 2)


               
            #predict pose
            if config.predict_pose:
                dec_pose = decoder_pose(enc_outputs['poses'], config)
                outputs.update(dec_pose)

            # all_scaling_factors = None
            outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)
            # outputs['all_rgb'] = None
            # all_scaling_factors = outputs['scaling_factor']
            # outputs['all_scaling_factors'] = all_scaling_factors
            # outputs['focal_length'] = None

            # # assemble net
            # outputs['transform'] = assembel(outputs['points_1'], outputs).unsqueeze(1)
            # outputs['pc_part_1_after'] = outputs['pc_part_1_pre'] * outputs['transform'][:, :, 0:3] + outputs['transform'][:, :, 3:6]
            # outputs['pc_part_2_after'] = outputs['pc_part_2_pre'] * outputs['transform'][:, :, 6:9] + outputs['transform'][:, :, 9:12]
            # outputs['pc_part_3_after'] = outputs['pc_part_3_pre'] * outputs['transform'][:, :, 12:15] + outputs['transform'][:, :, 15:18]
            # outputs['pc_part_4_after'] = outputs['pc_part_4_pre'] * outputs['transform'][:, :, 18:21] + outputs['transform'][:, :, 21:24]

            # outputs['points_1'] = torch.cat(( outputs['pc_part_1_after'], outputs['pc_part_2_after'], outputs['pc_part_3_after'], outputs['pc_part_4_after'] ), 1)


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
            task_loss = torch.nn.MSELoss(reduction='mean').cuda()
            if num_candidates >1 :
                gt_tensor, pred_tensor, min_loss = proj_loss_pose_candidates(IN_Mask, outputs['projs'], inputs, config)
                proj_loss = task_loss(gt_tensor, pred_tensor)/(config.batch_size)
                if config.pose_predictor_student:
                    student_loss = add_student_loss(inputs, outputs, min_loss, config)/(config.batch_size) * config.pose_predictor_student_loss_weight
                    total_loss = student_loss + proj_loss
            else:
                total_loss = task_loss(IN_Mask,outputs['projs']) * 4 * config.batch_size
            # gen_loss = (g1_loss +g2_loss +g3_loss +g4_loss +gfull_loss ) /(config.batch_size) *8 *5 
            gen_loss = g1_loss  +g2_loss   + g3_loss  + g4_loss
            loss = total_loss *4  + gen_loss
            # loss = total_loss


            loss.backward(retain_graph=True)
            if config.predict_pose:
                decoder_pose_opt.step()
            deback_opt.step()
            deseat_opt.step()
            delegs_opt.step()
            dearm_opt.step()
            encoder_img_opt.step()  
            # assembel_opt.step()

            
            writer.add_scalar('Loss/loss', loss.item(), n_iter)
            writer.add_scalar('Loss/recon_loss', total_loss.item(), n_iter)
            writer.add_scalar('Loss/dis_loss', dis_loss.item(), n_iter)
            writer.add_scalar('Loss/gen_loss', gen_loss.item(), n_iter)            
            
            print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)), 'epock: %d, iteration: %d/%d, loss: %f, dis_loss: %f, gen_loss: %f' %(epoch+1, 1+batch_idx, len(train_iter), loss, dis_loss, gen_loss))

        if epoch%10 == 0:
            torch.save(encoder_img, 'encoder_img.pkl')
            torch.save(deback, 'deback.pkl')
            torch.save(deseat, 'deseat.pkl')
            torch.save(delegs, 'delegs.pkl')
            torch.save(dearm, 'dearm.pkl')
            torch.save(discri_part_1, 'discri_part_1.pkl')
            torch.save(discri_part_2, 'discri_part_2.pkl')
            torch.save(discri_part_3, 'discri_part_3.pkl')
            torch.save(discri_part_4, 'discri_part_4.pkl')
            # torch.save(discri_full, 'discri_full.pkl')


        # if epoch%10==0:
        
            # torch.save(encoder_img, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_encoder'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            # torch.save(decoder_pc_part, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_decoderpc_part'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            # if config.predict_pose:
            #     torch.save(decoder_pose, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_posedec'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            # torch.save(discri_part_1, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_1'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_2, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_2'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_3, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_3'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_4, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_4'+str(round(total_loss.item(),2))+'.pkl')
       
            
    print('Finished Traning')   
    torch.save(encoder_img, snapshot_folder+'//'+'encoder_img_final.pkl')
    # torch.save(decoder_pc_part, snapshot_folder+'//'+'decoder_pc_final.pkl')
    if config.predict_pose:
        torch.save(decoder_pose, snapshot_folder+'//'+'decoder_pose_final.pkl')       
    torch.save(discri_part_1, snapshot_folder+'//'+'_discri_part_1'+'_final.pkl')
    torch.save(discri_part_2, snapshot_folder+'//'+'_discri_part_2'+'_final.pkl')
    torch.save(discri_part_3, snapshot_folder+'//'+'_discri_part_3'+'_final.pkl')
    torch.save(discri_part_4, snapshot_folder+'//'+'_discri_part_4'+'_final.pkl')
    writer.close()                


        
if __name__ == "__main__":
    train()


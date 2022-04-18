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
from pointnet import PointNetCls
from pc_projection import pointcloud_project, compute_projection
from pose_decoder import PoseDecoder
import torch.nn.functional as F
from scipy.io import savemat
from model_base import pool_single_view, preprocess
from model_pc import setup_sigma, replicate_for_multiview, tf_repeat_0, proj_loss_pose_candidates, add_student_loss
from torch.autograd import grad as torch_grad

from util.app_config import config as app_config
from util.train import get_trainable_variables, get_learning_rate
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression

from PIL import Image

# def adjust_learning_rate(optimizer, epoch, config, t=10):
#     """Sets the learning rate to the initial LR decayed by 10 every t epochs，default=10"""
#     new_lr = config.learning_rate * (0.1 ** (epoch // t))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = new_lr

def train():
    num_step = 0 
    critic_train_iteration = 5

    config = app_config
    config.cuda = not config.no_cuda
   # if config.gpu<0 and config.cuda:
    config.gpu = 2
    torch.cuda.set_device(config.gpu)
    if config.cuda and torch.cuda.is_available():
        print("Using CUDA on GPU ", config.gpu)
    else:
        print("Not using CUDA")

    print("Get model ... ... ")
    
    encoder_img = ImgEncoder()
    decoder_pc_part = PCDecoder_Part(config)
    # discri_part_1 = Discriminator(config)
    # discri_part_2 = Discriminator(config)
    # discri_part_3 = Discriminator(config)
    # discri_part_4 = Discriminator(config)
    # discri_full = Discriminator(config)
    discri_part_1 = PointNetCls()
    discri_part_2 = PointNetCls()
    discri_part_3 = PointNetCls()
    discri_part_4 = PointNetCls()
    discri_full = PointNetCls()

    if config.predict_pose:
        decoder_pose = PoseDecoder()

    encoder_img_opt = torch.optim.Adam(encoder_img.parameters(), lr = 1e-4, weight_decay = 1e-6)
    decoder_pc_part_opt = torch.optim.Adam(decoder_pc_part.parameters(), lr = 1e-4, weight_decay = 1e-6)
    if config.predict_pose:
        decoder_pose_opt = torch.optim.Adam(decoder_pose.parameters(), lr = 1e-4,weight_decay = 1e-6) 
    discri_part_1_opt = torch.optim.Adam(discri_part_1.parameters(), lr = 1e-7, weight_decay = 1e-8)
    discri_part_2_opt = torch.optim.Adam(discri_part_2.parameters(), lr = 1e-7, weight_decay = 1e-8)
    discri_part_3_opt = torch.optim.Adam(discri_part_3.parameters(), lr = 1e-7, weight_decay = 1e-8)
    discri_part_4_opt = torch.optim.Adam(discri_part_4.parameters(), lr = 1e-7, weight_decay = 1e-8)
    discri_full_opt = torch.optim.Adam(discri_full.parameters(), lr = 1e-7, weight_decay = 1e-8)
   
    train_dir = config.checkpoint_dir
    if not os.path.exists(train_dir):
        os.mknod(train_dir)     
    else:

        encoder_img= torch.load('encoder_img.pkl')
        decoder_pc_part = torch.load('decoder_pc_part.pkl')
        discri_part_1 = torch.load('discri_part_1.pkl')
        discri_part_2 = torch.load('discri_part_2.pkl')
        discri_part_3 = torch.load('discri_part_3.pkl')
        discri_part_4 = torch.load('discri_part_4.pkl')
        discri_full = torch.load('discri_full.pkl')
        encoder_img.train()
        decoder_pc_part.train()
        discri_part_1.train()
        discri_part_2.train()
        discri_part_3.train()
        discri_part_4.train()
        discri_full.train()

  
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

    # if config.save_log:
    #     fd_log = open('training_log.log', mode='a')
    #     fd_log.write('\n\nTraining log at '+ datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #     fd_log.write('\nepoch: {}'.format(config.epochs))
    #     fd_log.write('\nbatch_size: {}'.format(config.batch_size))
    #     fd_log.write('\ncuda: {}'.format(config.cuda))
    #     fd_log.flush()

    
    total_iter = config.epochs * len(train_iter)

    config.no_plot = False
    if not config.no_plot:
        plot_x = [x for x in range(total_iter)]
        plot_loss = [None for x in range(total_iter)]
        plot_rloss = [None for x in range(total_iter)]
        plot_geloss = [None for x in range(total_iter)]
        plot_dloss = [None for x in range(total_iter)]
        dyn_plot = DynamicPlot(title='Training loss over epochs (I2PC_part)', xdata=plot_x, ydata={'loss':plot_loss, 'dis_loss': plot_dloss, 'gen_loss': plot_geloss, 'recon_loss': plot_rloss })
        iter_id = 0
        max_loss = 0
    
    is_training = True
    outputs = {}
    criterion = torch.nn.BCELoss(reduction='sum')
    raw_inputs = {}
    params = {}
    params['_gauss_sigma'] = None
    params['_gauss_kernel'] = None
    params['_sigma_rel'] = None
    params['_alignment_to_canonical'] =None

    train_discri = True
  
    # criterion3 = torch.nn.BCELoss(reduction='mean')
    # criterion4 = torch.nn.BCELoss(reduction='mean')
    # criterion5 = torch.nn.BCELoss(reduction='mean')
    def gradient_penalty1(real_pc_part, gene_pc_part):
        batch_size=real_pc_part.size()[0]

         #calculate interpolation
        alpha1 = torch.rand(batch_size,1,1)
        alpha1 = alpha1.expand_as(real_pc_part)

        if config.cuda:
            alpha1=alpha1.cuda()

        interpolated1 = alpha1 * real_pc_part.data + (1 - alpha1) * gene_pc_part.data
        interpolated1 = Variable (interpolated1, requires_grad = True)

        if config.cuda:
            interpolated1 = interpolated1.cuda()
    
        #calculate probability of interpolated examples
        prob_interpolated = discri_part_1(interpolated1)

        #calculate gradients of probabilities with respect to examples
        gradients_pc = torch_grad(outputs=prob_interpolated, inputs = interpolated1,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if config.cuda else torch.ones( 
                                prob_interpolated.size()),
                                create_graph= True, retain_graph=True)[0]
                        
        #gradients have shape(batch_size, num_channels, img
        #so flatten to easily take norm per example in batc
        gradients_pc = gradients_pc.view(batch_size,-1)

        gradient_norm_pc  = torch.sqrt(torch.sum(gradients_pc ** 2, dim=1)+1e-12)
        return 10*((gradient_norm_pc - 1)** 2).mean()

    def gradient_penalty2(real_pc_part, gene_pc_part):
        batch_size=real_pc_part.size()[0]

         #calculate interpolation
        alpha1 = torch.rand(batch_size,1,1)
        alpha1 = alpha1.expand_as(real_pc_part)

        if config.cuda:
            alpha1=alpha1.cuda()

        interpolated1 = alpha1 * real_pc_part.data + (1 - alpha1) * gene_pc_part.data
        interpolated1 = Variable (interpolated1, requires_grad = True)

        if config.cuda:
            interpolated1 = interpolated1.cuda()
    
        #calculate probability of interpolated examples
        prob_interpolated = discri_part_2(interpolated1)

        #calculate gradients of probabilities with respect to examples
        gradients_pc = torch_grad(outputs=prob_interpolated, inputs = interpolated1,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if config.cuda else torch.ones( 
                                prob_interpolated.size()),
                                create_graph= True, retain_graph=True)[0]
                        
        #gradients have shape(batch_size, num_channels, img
        #so flatten to easily take norm per example in batc
        gradients_pc = gradients_pc.view(batch_size,-1)

        gradient_norm_pc  = torch.sqrt(torch.sum(gradients_pc ** 2, dim=1)+1e-12)

        return 10*((gradient_norm_pc - 1)** 2).mean()
    
    def gradient_penalty3(real_pc_part, gene_pc_part):
        batch_size=real_pc_part.size()[0]

         #calculate interpolation
        alpha1 = torch.rand(batch_size,1,1)
        alpha1 = alpha1.expand_as(real_pc_part)

        if config.cuda:
            alpha1=alpha1.cuda()

        interpolated1 = alpha1 * real_pc_part.data + (1 - alpha1) * gene_pc_part.data
        interpolated1 = Variable (interpolated1, requires_grad = True)

        if config.cuda:
            interpolated1 = interpolated1.cuda()
    
        #calculate probability of interpolated examples
        prob_interpolated = discri_part_3(interpolated1)

        #calculate gradients of probabilities with respect to examples
        gradients_pc = torch_grad(outputs=prob_interpolated, inputs = interpolated1,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if config.cuda else torch.ones( 
                                prob_interpolated.size()),
                                create_graph= True, retain_graph=True)[0]
                        
        #gradients have shape(batch_size, num_channels, img
        #so flatten to easily take norm per example in batc
        gradients_pc = gradients_pc.view(batch_size,-1)

        gradient_norm_pc  = torch.sqrt(torch.sum(gradients_pc ** 2, dim=1)+1e-12)

        return 10*((gradient_norm_pc - 1)** 2).mean()
    
    def gradient_penalty4(real_pc_part, gene_pc_part):
        batch_size=real_pc_part.size()[0]

         #calculate interpolation
        alpha1 = torch.rand(batch_size,1,1)
        alpha1 = alpha1.expand_as(real_pc_part)

        if config.cuda:
            alpha1=alpha1.cuda()

        interpolated1 = alpha1 * real_pc_part.data + (1 - alpha1) * gene_pc_part.data
        interpolated1 = Variable (interpolated1, requires_grad = True)

        if config.cuda:
            interpolated1 = interpolated1.cuda()
    
        #calculate probability of interpolated examples
        prob_interpolated = discri_part_4(interpolated1)

        #calculate gradients of probabilities with respect to examples
        gradients_pc = torch_grad(outputs=prob_interpolated, inputs = interpolated1,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if config.cuda else torch.ones( 
                                prob_interpolated.size()),
                                create_graph= True, retain_graph=True)[0]
                        
        #gradients have shape(batch_size, num_channels, img
        #so flatten to easily take norm per example in batc
        gradients_pc = gradients_pc.view(batch_size,-1)

        gradient_norm_pc  = torch.sqrt(torch.sum(gradients_pc ** 2, dim=1)+1e-12)

        return 10*((gradient_norm_pc - 1)** 2).mean()

    def gradient_penalty_full(real_pc_part, gene_pc_part):
        batch_size=real_pc_part.size()[0]

         #calculate interpolation
        alpha1 = torch.rand(batch_size,1,1)
        alpha1 = alpha1.expand_as(real_pc_part)

        if config.cuda:
            alpha1=alpha1.cuda()

        interpolated1 = alpha1 * real_pc_part.data + (1 - alpha1) * gene_pc_part.data
        interpolated1 = Variable (interpolated1, requires_grad = True)

        if config.cuda:
            interpolated1 = interpolated1.cuda()
    
        #calculate probability of interpolated examples
        prob_interpolated = discri_full(interpolated1)

        #calculate gradients of probabilities with respect to examples
        gradients_pc = torch_grad(outputs=prob_interpolated, inputs = interpolated1,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if config.cuda else torch.ones( 
                                prob_interpolated.size()),
                                create_graph= True, retain_graph=True)[0]
                        
        #gradients have shape(batch_size, num_channels, img
        #so flatten to easily take norm per example in batc
        gradients_pc = gradients_pc.view(batch_size,-1)

        gradient_norm_pc  = torch.sqrt(torch.sum(gradients_pc ** 2, dim=1)+1e-12)

        return 10*((gradient_norm_pc - 1)** 2).mean()

    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(train_iter):
            #prepare the parameters such as kernal and sigma
            params['_global_step'] = epoch * len(train_iter) + batch_idx 
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

            encoder_img_opt.zero_grad()
            decoder_pc_part_opt.zero_grad()
            if config.predict_pose:
                decoder_pose_opt.zero_grad()

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

            #build Decoder
            dec_pcoutputs = decoder_pc_part(outputs['ids_1'],config)
            outputs.update(dec_pcoutputs) 

            if train_discri:
            # for s in range(0,2):

                #part 1 back
                real_out1 = discri_part_1(IN_PC_Part1)              
                fake_out1 = discri_part_1(outputs['pc_part_1'].detach() * 2) 
                
                #get gradient panalty
                loss_gp_1 = gradient_penalty1(IN_PC_Part1, outputs['pc_part_1'].detach() * 2)  
                d1_loss = fake_out1.mean() - real_out1.mean() + loss_gp_1
                        

                #part 2 
                real_out2 = discri_part_2(IN_PC_Part2)               
                fake_out2 = discri_part_2(outputs['pc_part_2'].detach() * 2)
                loss_gp_2 = gradient_penalty2(IN_PC_Part2, outputs['pc_part_2'].detach() * 2)  
                d2_loss = fake_out2.mean() - real_out2.mean() + loss_gp_2           
                               

                #part 3 leg
                real_out3 = discri_part_3(IN_PC_Part3)
                fake_out3 = discri_part_3(outputs['pc_part_3'].detach() * 2)
                loss_gp_3 = gradient_penalty3(IN_PC_Part3, outputs['pc_part_3'].detach() * 2)  
                d3_loss = fake_out3.mean() - real_out3.mean() + loss_gp_3               


                #part 4 arm
                real_out4  = discri_part_4(IN_PC_Part4)
                fake_out4 = discri_part_4(outputs['pc_part_4'].detach() * 2)
                loss_gp_4 = gradient_penalty4(IN_PC_Part4, outputs['pc_part_4'].detach() * 2)  
                d4_loss = fake_out4.mean() - real_out4.mean() + loss_gp_4

		        #connect to full shape
                outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)

                #part full 
                real_out_full  = discri_full(IN_PC_Full)
                fake_out_full = discri_full(outputs['points_1'].detach() * 2)
                loss_gp_full = gradient_penalty_full(IN_PC_Full, outputs['points_1'].detach() * 2)  
                dfull_loss = fake_out_full.mean() - real_out_full.mean() + loss_gp_full         
            

                dis_loss  = (d1_loss + d2_loss + d3_loss + d4_loss + dfull_loss)/(config.batch_size) *8 *5
                #dis_loss =  (d1_loss + d2_loss + d3_loss + d4_loss )/(config.batch_size) *8 *5
                # torch.cuda.empty_cache() 
                dis_loss.backward(retain_graph=True)
                
                discri_part_2_opt.step()
                discri_part_3_opt.step()
                discri_part_4_opt.step()
                discri_part_1_opt.step()
                discri_full_opt.step()
                # dis_accuracy =torch.sum( real_out1 + real_out2 + real_out3 + real_out4 )/ (config.batch_size * 4) + \
                #                 (1 - torch.sum(fake_out1 +fake_out2 +fake_out3 +fake_out4) / (config.batch_size * 4))
                # if dis_accuracy > 1.6:
                #     train_discri = False

            #generate_loss
            fake_out_g1 = discri_part_1(outputs['pc_part_1'] * 2)
            fake_out_g2 = discri_part_2(outputs['pc_part_2'] * 2)
            fake_out_g3 = discri_part_3(outputs['pc_part_3'] * 2)
            fake_out_g4 = discri_part_4(outputs['pc_part_4'] * 2)
            fake_out_gfull = discri_full(outputs['points_1'] * 2)
            g1_loss = fake_out_g1.mean()
            g2_loss = fake_out_g2.mean()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            g3_loss = fake_out_g3.mean()
            g4_loss = fake_out_g4.mean()
            gfull_loss = fake_out_gfull.mean()
            # gfull_loss = criterion(fake_out_gfull, real_label_full)
            # gen_accuracy =torch.sum( fake_out_g1 + fake_out_g2 + fake_out_g3 + fake_out_g4 )/ (config.batch_size * 4)
            # print(gen_accuracy)
            # if gen_accuracy > 0.7:
            #     train_discri = True
            #predict pose
            if config.predict_pose:
                dec_pose = decoder_pose(enc_outputs['poses'], config)
                outputs.update(dec_pose)

            # all_scaling_factors = None
            outputs['points_1'] =  torch.cat((outputs['pc_part_1'],outputs['pc_part_2'],outputs['pc_part_3'],outputs['pc_part_4']),1)
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

            #loss
            IN_Mask = inputs['masks'].permute(0,3,1,2)
            IN_Mask = F.interpolate(IN_Mask,[64,64])
            IN_Mask = IN_Mask.permute(0,2,3,1)
            task_loss = torch.nn.MSELoss(reduction='sum').cuda()
            if num_candidates >1 :
                gt_tensor, pred_tensor, min_loss = proj_loss_pose_candidates(IN_Mask, outputs['projs'], inputs, config)
                proj_loss = task_loss(gt_tensor, pred_tensor)/(config.batch_size)
                if config.pose_predictor_student:
                    student_loss = add_student_loss(inputs, outputs, min_loss, config)/(config.batch_size) * config.pose_predictor_student_loss_weight
                    total_loss = student_loss + proj_loss
            else:
                total_loss = task_loss(IN_Mask, outputs['projs'])/(config.batch_size)     
            gen_loss = (g1_loss  +g2_loss   + g3_loss  + g4_loss + gfull_loss)/(config.batch_size) *8 *5
            loss = total_loss + gen_loss
            # loss = total_loss
            loss.backward(retain_graph=True)
            if config.predict_pose:
                decoder_pose_opt.step()
            decoder_pc_part_opt.step()
            encoder_img_opt.step()  

            # for paramss in discri_part_1.named_parameters():
            #     [name, params1] = paramss
            #     if "weight" in name:
            #         print(name, ':', params1.grad.data.mean())

            # for paramss in discri_part_2.named_parameters():
            #     [name, params1] = paramss
            #     if "weight" in name:
            #         print(name, ':2', params1.grad.data.mean())

            if not config.no_plot:
                plot_loss[iter_id] = loss.item()
                plot_rloss[iter_id] = total_loss.item()
                # gen_loss = loss
                # dis_loss =loss
                plot_geloss[iter_id] = gen_loss.item()*20
                plot_dloss[iter_id] = dis_loss.item()*0.1
                max_loss = max(max_loss,loss.item())
                dyn_plot.setxlim(0., (iter_id+1)*1.05)
                dyn_plot.setylim(0., max_loss*1.05)
                dyn_plot.update_plots(ydata = {'loss':plot_loss, 'dis_loss': plot_dloss, 'gen_loss': plot_geloss, 'recon_loss': plot_rloss})
                iter_id += 1
            
            
            
            print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)), 'epock: %d, iteration: %d/%d, loss: %f, dis_loss: %f, gen_loss: %f' %(epoch+1, 1+batch_idx, len(train_iter), loss.item(), dis_loss.item(), gen_loss.item()))


        if (epoch+1)%10 == 0:
            plt.savefig(str(epoch)+'.jpg')
            torch.save(encoder_img, 'encoder_img.pkl')
            torch.save(decoder_pc_part, 'decoder_pc_part.pkl')
            torch.save(discri_part_1, 'discri_part_1.pkl')
            torch.save(discri_part_2, 'discri_part_2.pkl')
            torch.save(discri_part_3, 'discri_part_3.pkl')
            torch.save(discri_part_4, 'discri_part_4.pkl')
            torch.save(discri_full, 'discri_full.pkl')

        # if epoch%10==0:
        
            torch.save(encoder_img, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_encoder'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            torch.save(decoder_pc_part, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_decoderpc_part'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            if config.predict_pose:
                torch.save(decoder_pose, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_posedec'+str(round(total_loss.item(),2))+'_epoch'+str(epoch)+'.pkl')
            # torch.save(discri_part_1, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_1'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_2, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_2'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_3, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_3'+str(round(total_loss.item(),2))+'.pkl')
            # torch.save(discri_part_4, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_discri_part_4'+str(round(total_loss.item(),2))+'.pkl')
       
            
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
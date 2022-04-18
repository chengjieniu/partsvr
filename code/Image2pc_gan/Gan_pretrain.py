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
from torch.autograd import grad as torch_grad

from util.app_config import config as app_config
from util.train import get_trainable_variables, get_learning_rate
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression

from PIL import Image



def train():

    num_step = 0 
    critic_train_iteration = 5
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
    
    decoder_pc_part = PCDecoder_Part(config)
    discri_part_1 = Discriminator(config)
    discri_part_2 = Discriminator(config)
    discri_part_3 = Discriminator(config)
    discri_part_4 = Discriminator(config)
    # discri_full = Discriminator(config)
 


    decoder_pc_part_opt = torch.optim.Adam(decoder_pc_part.parameters(), lr = 1e-3, weight_decay = 1e-6)
    discri_part_1_opt = torch.optim.Adam(discri_part_1.parameters(), lr = 1e-5, weight_decay = 1e-6)
    discri_part_2_opt = torch.optim.Adam(discri_part_2.parameters(), lr = 1e-5, weight_decay = 1e-6)
    discri_part_3_opt = torch.optim.Adam(discri_part_3.parameters(), lr = 1e-5, weight_decay = 1e-6)
    discri_part_4_opt = torch.optim.Adam(discri_part_4.parameters(), lr = 1e-5, weight_decay = 1e-6)
    # discri_full_opt = torch.optim.Adam(discri_full.parameters(), lr = 1e-4, weight_decay = 1e-6)

    # train_dir = config.checkpoint_dir
    # if not os.path.exists(train_dir):
    #     os.mknod(train_dir)     
    # else:
    #     checkpoint = torch.load(train_dir)
    #     encoder_img.load_state_dict(checkpoint['encoder_img_state_dict'])
    #     decoder_pc.load_state_dict(checkpoint['decoder_pc_state_dict'])
    #     decoder_pose.load_state_dict(checkpoint['decoder_pose_state_dict'])
  
    if config.cuda:

        decoder_pc_part.cuda()
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

    total_iter = config.epochs * len(train_iter)
    if config.save_snapshot:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        snapshot_folder = os.path.join(config.save_path, 'snapshots_'+ strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        if not os.path.exists(snapshot_folder):
            os.makedirs(snapshot_folder)
    
    config.no_plot = True
    if not config.no_plot:
        plot_x = [x for x in range(total_iter)]
        plot_loss = [None for x in range(total_iter)]
        plot_gloss = [None for x in range(total_iter)]
        dyn_plot = DynamicPlot(title='Training loss over epochs (I2PC_part)', xdata=plot_x, ydata={'loss':plot_loss, 'gan_loss': plot_gloss})
        iter_id = 0
        max_loss = 0
    
    is_training = True
    outputs = {}
    criterion = torch.nn.BCELoss(reduction='sum')
    train_discri = True
  
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

    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(train_iter):
            #prepare the parameters such as kernal and sigma
             
              
            IN_PC_Part1 = [x[4] for x in batch]
            IN_PC_Part2 = [x[5] for x in batch]
            IN_PC_Part3 = [x[6] for x in batch]
            IN_PC_Part4 = [x[7] for x in batch]
         
            IN_PC_Part1 = torch.stack(IN_PC_Part1)
            IN_PC_Part2 = torch.stack(IN_PC_Part2)
            IN_PC_Part3 = torch.stack(IN_PC_Part3)
            IN_PC_Part4 = torch.stack(IN_PC_Part4)
                        
            IN_PC_Part1 = utiltorch.var_or_cuda(IN_PC_Part1)
            IN_PC_Part2 = utiltorch.var_or_cuda(IN_PC_Part2)
            IN_PC_Part3 = utiltorch.var_or_cuda(IN_PC_Part3)
            IN_PC_Part4 = utiltorch.var_or_cuda(IN_PC_Part4)
            
            decoder_pc_part_opt.zero_grad()
            discri_part_1_opt.zero_grad()
            discri_part_2_opt.zero_grad()
            discri_part_3_opt.zero_grad()
            discri_part_4_opt.zero_grad()

            outputs['ids_1'] = Variable(decoder_pc_part.sample_latent(config.batch_size)).cuda()
            #build Decoder
            # if num_step % critic_train_iteration == 0:
            dec_pcoutputs = decoder_pc_part(outputs['ids_1'],config)
            outputs.update(dec_pcoutputs) 
            

            # if train_discri:
            if num_step % critic_train_iteration == 0:

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
                
        
                gan_loss  = (d1_loss + d2_loss + d3_loss + d4_loss)
                gan_loss.backward(retain_graph=True)
                
                discri_part_2_opt.step()
                discri_part_3_opt.step()
                discri_part_4_opt.step()
                discri_part_1_opt.step()
               
            # if num_step % critic_train_iteration == 0:

                #generate_loss
            fake_out_g1 = discri_part_1(outputs['pc_part_1'] * 2)
            fake_out_g2 = discri_part_1(outputs['pc_part_2'] * 2)
            fake_out_g3 = discri_part_1(outputs['pc_part_3'] * 2)
            fake_out_g4 = discri_part_1(outputs['pc_part_4'] * 2)
                # fake_out_gfull = discri_full(outputs['points_1'] * 2)
            g1_loss = fake_out_g1.mean()
            g2_loss = fake_out_g2.mean()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            g3_loss = fake_out_g3.mean()
            g4_loss = fake_out_g4.mean()

    
            loss = (g1_loss +g2_loss +g3_loss +g4_loss)         

            loss.backward(retain_graph=True)

            decoder_pc_part_opt.step()
 

            if not config.no_plot:
                plot_loss[iter_id] = loss.item()
                plot_gloss[iter_id] = gan_loss.item()
                max_loss = max(max_loss,loss.item()*10)
                dyn_plot.setxlim(0., (iter_id+1)*1.05)
                dyn_plot.setylim(-40., max_loss*1.05)
                dyn_plot.update_plots(ydata = {'loss':plot_loss, 'gan_loss': plot_gloss})
                iter_id += 1
            num_step += 1
            
            print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)), 'epock: %d, iteration: %d/%d, loss: %f, gan_loss: %f' %(epoch+1, 1+batch_idx, len(train_iter), loss, gan_loss))

        if epoch%10==0:
        
            
            torch.save(decoder_pc_part, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_decoderpc_part'+str(round(loss.item(),2))+'.pkl')
            
            
    print('Finished Traning')   
    torch.save(decoder_pc_part, snapshot_folder+'//'+'decoder_pc_final.pkl')
              


        
if __name__ == "__main__":
    train()


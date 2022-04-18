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
from mydataset import myDataset
from img_encoder import ImgEncoder
from pc_decoder import PCDecoder
from pc_projection import pointcloud_project, predict_focal_length, predict_scaling_factor, compute_projection
from pose_decoder import PoseDecoder
import torch.nn.functional as F
from scipy.io import savemat

from util.app_config import config as app_config
from util.train import get_trainable_variables, get_learning_rate
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression

from tensorboardX import SummaryWriter

from PIL import Image

writer = SummaryWriter()
def train():
    # p1 = torch.randn(1,4)
    # p2 = torch.randn(1,8000,3)

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
    # decoder_pc = PCDecoder()
    decoder_pose = PoseDecoder()

    encoder_img_opt = torch.optim.Adam(encoder_img.parameters(), lr = 1e-4, weight_decay = 1e-3)
    # decoder_pc_opt = torch.optim.Adam(decoder_pc.parameters(), lr = 1e-4,  weight_decay = 1e-3)
    decoder_pose_opt = torch.optim.Adam(decoder_pose.parameters(), lr = 1e-2,weight_decay = 1e-1) 

    # train_dir = config.checkpoint_dir
    # if not os.path.exists(train_dir):
    #     os.mknod(train_dir)     
    # else:
    #     checkpoint = torch.load(train_dir)
    #     encoder_img.load_state_dict(checkpoint['encoder_img_state_dict'])
    #     decoder_pc.load_state_dict(checkpoint['decoder_pc_state_dict'])
    #     decoder_pose.load_state_dict(checkpoint['decoder_pose_state_dict'])
  
    if config.cuda:
        encoder_img.cuda()
        # decoder_pc.cuda()
        decoder_pose.cuda()

    print("Loading data ... ... ", end = '', flush = True)

    data_img2pc = myDataset(config.data_path)
    def my_collate(batch):
        return batch
    train_iter = torch.utils.data.DataLoader(data_img2pc, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
  

    print("Start training ... ...")

    start = time.time()

    # if config.save_snapshot:
    #     if not os.path.exists(config.save_path):
    #         os.makedirs(config.save_path)
    #     snapshot_folder = os.path.join(config.save_path, 'snapshots_'+ strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
    #     if not os.path.exists(snapshot_folder):
    #         os.makedirs(snapshot_folder)

    # if config.save_log:
    #     fd_log = open('training_log.log', mode='a')
    #     fd_log.write('\n\nTraining log at '+ datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #     fd_log.write('\nepoch: {}'.format(config.epochs))
    #     fd_log.write('\nbatch_size: {}'.format(config.batch_size))
    #     fd_log.write('\ncuda: {}'.format(config.cuda))
    #     fd_log.flush()
    
    total_iter = config.epochs * len(train_iter)

    # if not config.no_plot:
    #     plot_x = [x for x in range(total_iter)]
    #     plot_loss = [None for x in range(total_iter)]
    #     dyn_plot = DynamicPlot(title='Training loss over epochs (I2PC)', xdata=plot_x, ydata={'loss':plot_loss})
    #     iter_id = 0
    #     max_loss = 0
    
    is_training = True
    outputs = {}
    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(train_iter):
            
            IN_Image = [x[0] for x in batch]
            IN_Mask = [x[1] for x in batch]

            # IN_Image = train_iter.dataset[0].unsqueeze(0)
            # IN_Mask = train_iter.dataset[1].unsqueeze(0)

            IN_Image = torch.stack(IN_Image)
            IN_Mask = torch.stack(IN_Mask)
            IN_Image = utiltorch.var_or_cuda(IN_Image)
            IN_Mask = utiltorch.var_or_cuda(IN_Mask)
            IN_pc = [x[2]for x in batch]
            IN_pc = torch.stack(IN_pc)
            IN_pc = utiltorch.var_or_cuda(IN_pc)

            encoder_img_opt.zero_grad()
            # decoder_pc_opt.zero_grad()
            decoder_pose_opt.zero_grad()

            #build Encoder
            enc_outputs = encoder_img(IN_Image.unsqueeze(0))
            ids = enc_outputs['ids']
            outputs['conv_features'] = enc_outputs['conv_features']
            outputs['ids'] = ids
            outputs['z_latent'] = enc_outputs['z_latent']
            outputs['ids_1'] = ids

            #build Decoder
            # dec_pcoutputs = decoder_pc( outputs['ids_1'],config)
            # outputs.update(dec_pcoutputs)
       
            # outputs['scaling_factor'] = predict_scaling_factor(config, outputs['ids_1'], is_training)
            # outputs['focal_length'] = predict_focal_length( config, outputs['ids'], is_training)

            #predict pose
            dec_pose = decoder_pose(enc_outputs['poses'])
            outputs.update(dec_pose)
            
            # outputs['poses'] =IN_quat

            # all_scaling_factors = None
            outputs['points_1'] =IN_pc
            outputs['all_rgb'] = None
            # all_scaling_factors = outputs['scaling_factor']
            # outputs['all_scaling_factors'] = all_scaling_factors
            outputs['all_scaling_factors'] = None
            outputs['focal_length'] = None
            proj_output = compute_projection( outputs)
            outputs.update(proj_output)

            for params in encoder_img.named_parameters():
                [name, param] = params
                if "weight" in name:
                    print(name, ':', param.data.mean())

            # for params in decoder_pc.named_parameters():
            #     [name, param] = params
            #     if "weight" in name:
            #         print(name, ':', param.data.mean())
            # print("^^^^^^^^^^^")
            

            # if batch_idx%2 ==0:
            #     for name, param in encoder_img.named_parameters():
            #         writer.add_histogram(name, param.clone().cpu().data.numpy(), batch_idx)
            #     for name, param in decoder_pc.named_parameters():
            #         writer.add_histogram(name, param.clone().cpu().data.numpy(), batch_idx)
            #     for name, param in decoder_pose.named_parameters():
            #         writer.add_histogram(name, param.clone().cpu().data.numpy(), batch_idx)

       
                    

            # Vox = torch.randn(1,64,64,1)
            # #pc1 = torch.randn(1,8000,3)
            # pose1 = torch.zeros(4,4)
            #loss
            IN_Mask = IN_Mask.permute(0,3,1,2)
            IN_Mask = F.interpolate(IN_Mask,[64,64])
            IN_Mask = IN_Mask.permute(0,2,3,1)
            task_loss = torch.nn.MSELoss(reduction='mean')
            loss = task_loss(IN_Mask,outputs['projs'])            

            loss.backward(retain_graph=True)

            # for params in encoder_img.named_parameters():
            #     [name, param] = params
            #     if "weight" in name:
            #         print(name, ':', param.grad.data.mean())
                    
                    
            decoder_pose_opt.step()
            # decoder_pc_opt.step()
            encoder_img_opt.step()  

            # if not config.no_plot:
            #     plot_loss[iter_id] = loss.item()
            #     max_loss = max(max_loss,loss.item())
            #     dyn_plot.setxlim(0., (iter_id+1)*1.05)
            #     dyn_plot.setylim(0., max_loss*1.05)
            #     dyn_plot.update_plots(ydata = {'loss':plot_loss})
            #     iter_id += 1
            
            print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)), 'epock: %d, iteration: %d/%d, loss: %f' %(epoch+1, 1+batch_idx, len(train_iter), loss))

            # if epoch%500==0:
            #     #save state_dict
            #     # torch.save(encoder_img.state_dict(),  train_dir)
            #     # torch.save(decoder_pc.state_dict(),  train_dir)
            #     # torch.save(decoder_pose.state_dict(),  train_dir)
            # #     # torch.save({
            # #     #             'epoch': epoch,
            # #     #             'encoder_img_state_dict': encoder_img.state_dict(),
            # #     #             'decoder_pc_state_dict':decoder_pc.state_dict(),
            # #     #             'decoder_pose_state_dict':decoder_pose.state_dict(),
            # #     #             'encoder_img_opt_state_dict': encoder_img_opt.state_dict(),
            # #     #             'decoder_pc_opt_state_dict': decoder_pc_opt.state_dict(),
            # #     #             'decoder_pose_opt_state_dict': decoder_pose_opt.state_dict(),
            # #     #             'loss': loss
            # #     #             }, train_dir)
            #     torch.save(encoder_img, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_'+str(round(loss.item(),2))+'.pkl')
            #     torch.save(decoder_pc, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_'+str(round(loss.item(),2))+'.pkl')
            #     torch.save(decoder_pose, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_'+str(round(loss.item(),2))+'.pkl')
              
    print('Finished Traning')   
    torch.save(encoder_img, snapshot_folder+'//'+'encoder_img_final.pkl')
    # torch.save(decoder_pc, snapshot_folder+'//'+'decoder_pc_final.pkl')
    torch.save(decoder_pose, snapshot_folder+'//'+'decoder_pose_final.pkl')                     
    writer.close()

        
if __name__ == "__main__":
    train()
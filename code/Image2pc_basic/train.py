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
from pc_projection import pointcloud_project,  compute_projection
from pose_decoder import PoseDecoder
import torch.nn.functional as F
from scipy.io import savemat
from model_base import pool_single_view

from util.app_config import config as app_config
from util.train import get_trainable_variables, get_learning_rate
from util.losses import regularization_loss
from util.fs import mkdir_if_missing
from util.data import tf_record_compression

from tensorboardX import SummaryWriter

from PIL import Image
from model_base import preprocess
from model_pc import setup_sigma, replicate_for_multiview, tf_repeat_0, proj_loss_pose_candidates, add_student_loss
#write
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
    # if config.gpu<0 and config.cuda:
    config.gpu = 0
    torch.cuda.set_device(config.gpu)
    if config.cuda and torch.cuda.is_available():
        print("Using CUDA on GPU ", config.gpu)
    else:
        print("Not using CUDA")

    print("Get model ... ... ")
    
    encoder_img = ImgEncoder()
    decoder_pc = PCDecoder()
    if config.predict_pose:
        decoder_pose = PoseDecoder()

    # encoder_img=torch.load(config.save_path+'//'+'encoder_img_final.pkl')
    # decoder_pc = torch.load(config.save_path+'//'+'decoder_pc_final.pkl')
    # encoder_img.eval()
    # decoder_pc.eval()

    encoder_img_opt = torch.optim.Adam(encoder_img.parameters(), lr = 1e-4, weight_decay = 1e-5)
    decoder_pc_opt = torch.optim.Adam(decoder_pc.parameters(), lr = 1e-4,  weight_decay = 1e-5)
    if config.predict_pose:
        decoder_pose_opt = torch.optim.Adam(decoder_pose.parameters(), lr = 1e-4,weight_decay = 1e-5) 

    # train_dir = config.checkpoint_dir
    # if not os.path.exists(train_dir):
    #     os.mknod(train_dir)
    # else:
    #     # checkpoint = torch.load(train_dir)
    #
    #     # encoder_img.load_state_dict(checkpoint['encoder_img_state_dict'])
    #     # decoder_pc.load_state_dict(checkpoint['decoder_pc_state_dict'])
    #     # decoder_pose.load_state_dict(checkpoint['decoder_pose_state_dict'])
    #     encoder_img = torch.load('encoder_img.pkl')
    #     decoder_pc = torch.load('decoder_pc.pkl')
    #     encoder_img.train()
    #     decoder_pc.train()


    if config.cuda:
        encoder_img.cuda()
        decoder_pc.cuda()
        if config.predict_pose:
            decoder_pose.cuda()

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
        snapshot_folder = os.path.join(config.save_path, 'snapshots_' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        if not os.path.exists(snapshot_folder):
            os.makedirs(snapshot_folder)


    # if config.save_log:
    #     fd_log = open('training_log.log', mode='a')
    #     fd_log.write('\n\nTraining log at '+ datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #     fd_log.write('\nepoch: {}'.format(config.epochs))
    #     fd_log.write('\nbatch_size: {}'.format(config.batch_size))
    #     fd_log.write('\ncuda: {}'.format(config.cuda))
    #     fd_log.flush()
    
    # total_iter = config.epochs * len(train_iter)

    # if not config.no_plot:
    #     plot_x = [x for x in range(total_iter)]
    #     plot_loss = [None for x in range(total_iter)]
    #     dyn_plot = DynamicPlot(title='Training loss over epochs (I2PC)', xdata=plot_x, ydata={'loss':plot_loss})
    #     iter_id = 0
    #     max_loss = 0
    
    is_training = True
    outputs = {}
    raw_inputs = {}
    params = {}
    params['_gauss_sigma'] = None
    params['_gauss_kernel'] = None
    params['_sigma_rel'] = None
    params['_alignment_to_canonical'] =None
    n_iter = 0

    for epoch in range(0 ,config.epochs):
        # adjust_learning_rate(encoder_img_opt, epoch, config, 10 )  
        # adjust_learning_rate(decoder_pc_opt, epoch, config, 10)  
        # if config.predict_pose:
        #     adjust_learning_rate(decoder_pose_opt, epoch, config, 5)  


        for batch_idx, batch in enumerate(train_iter):

            n_iter = n_iter+1
            #prepare the parameters such as kernal and sigma
            params['_global_step'] = epoch*len(train_iter) + batch_idx 
            params['_sigma_rel'], params['_gauss_sigma'], params['_gauss_kernel'] = setup_sigma(config, params['_global_step'], len(train_iter))
            # setup_misc()      
         
            IN_Image = [x[0] for x in batch]
            IN_Mask = [x[1] for x in batch]
            IN_Cameras = [x[2] for x in batch]
            IN_Cam_pos =[x[3] for x in batch]

            # IN_Image = train_iter.dataset[0].unsqueeze(0)
            # IN_Mask = train_iter.dataset[1].unsqueeze(0)

            IN_Image = torch.stack(IN_Image)
            IN_Mask = torch.stack(IN_Mask)
            IN_Cameras = torch.stack(IN_Cameras)
            IN_Cam_pos = torch.stack(IN_Cam_pos)
            raw_inputs['image'] = utiltorch.var_or_cuda(IN_Image)
            raw_inputs['mask'] = utiltorch.var_or_cuda(IN_Mask)
            raw_inputs['cameras'] = utiltorch.var_or_cuda(IN_Cameras)
            raw_inputs['cam_pos'] = utiltorch.var_or_cuda(IN_Cam_pos)
            # IN_quat = [x[2]for x in batch]
            # IN_quat = torch.stack(IN_quat)
            # IN_quat = utiltorch.var_or_cuda(IN_quat)
            
            inputs = preprocess(raw_inputs, config)

            encoder_img_opt.zero_grad()
            decoder_pc_opt.zero_grad()
            if config.predict_pose:
                decoder_pose_opt.zero_grad()

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
            # key = 'ids' if predict_for_all else 'ids_1'
            dec_pcoutputs = decoder_pc(outputs['ids_1'],config)
            outputs.update(dec_pcoutputs)
       

            #predict pose
            if config.predict_pose:
                dec_pose = decoder_pose(enc_outputs['poses'], config)
                outputs.update(dec_pose)
            
            
            
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

            # for params in encoder_img.named_parameters():
            #     [name, param] = params
            #     if "weight" in name:
            #         print(name, ':', param.data.mean())

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
            IN_Mask = inputs['masks'].permute(0,3,1,2)
            IN_Mask = F.interpolate(IN_Mask,[64,64])
            IN_Mask = IN_Mask.permute(0,2,3,1)
            task_loss = torch.nn.MSELoss(reduction='sum')
            if num_candidates >1 :
                gt_tensor, pred_tensor, min_loss = proj_loss_pose_candidates(IN_Mask, outputs['projs'], inputs, config)
                proj_loss = task_loss(gt_tensor, pred_tensor)/(config.batch_size)
                if config.pose_predictor_student:
                    student_loss = add_student_loss(inputs, outputs, min_loss, config)/(config.batch_size) * config.pose_predictor_student_loss_weight
                    total_loss = student_loss + proj_loss
            else:
                total_loss = task_loss(IN_Mask, outputs['projs'])/outputs['projs'].shape[0]
                


            total_loss.backward()

            # for params in encoder_img.named_parameters():
            #     [name, param] = params
            #     if "weight" in name:
            #         print(name, ':', param.grad.data.mean())
                    
                    
            if config.predict_pose:
                decoder_pose_opt.step()
            decoder_pc_opt.step()
            encoder_img_opt.step()  

            # if not config.no_plot:
            #     plot_loss[iter_id] = total_loss.item()
            #     max_loss = max(max_loss,total_loss.item())
            #     dyn_plot.setxlim(0., (iter_id+1)*1.05)
            #     dyn_plot.setylim(0., max_loss*1.05)
            #     dyn_plot.update_plots(ydata = {'loss':plot_loss})
            #     iter_id += 1
            
            # if total_loss < 700:
            #     plt.savefig(str(epoch)+'700h.jpg')

            writer.add_scalar('Loss', total_loss.item(), n_iter)
            print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)), 'epock: %d, iteration: %d/%d, loss: %f' %(epoch+1, 1+batch_idx, len(train_iter), total_loss))

        # if (epoch+1)%10 == 0:
        #     plt.savefig(str(epoch)+'.jpg')
        #     torch.save(encoder_img, 'encoder_img.pkl')
        #     torch.save(decoder_pc, 'decoder_pc.pkl')
        if (epoch+1)%100==0:

            torch.save(encoder_img, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_encoder_basic'+str(round(total_loss.item(),2))+'.pkl')
            torch.save(decoder_pc, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_decoder_basic'+str(round(total_loss.item(),2))+'.pkl')
            if config.predict_pose:
                torch.save(decoder_pose, snapshot_folder+'//'+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))+'_posedec_basic'+str(round(total_loss.item(),2))+'.pkl')
              
    print('Finished Traning')   
    torch.save(encoder_img, snapshot_folder+'//'+'encoder_img_final.pkl')
    torch.save(decoder_pc, snapshot_folder+'//'+'decoder_pc_final.pkl')
    if config.predict_pose:
        torch.save(decoder_pose, snapshot_folder+'//'+'decoder_pose_final.pkl')                     
    writer.close()

        
if __name__ == "__main__":
    train()      





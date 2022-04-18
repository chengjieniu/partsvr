import os
import h5py
import numpy as np
import scipy.io as scio
from scipy.io import loadmat
import torch
from show_pc import ShowPC
from scipy.io import loadmat
import show_balls
import matplotlib.pyplot as plt
print('load results')

# pred = loadmat('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/03001627pred_test_gan.mat')
# pred = loadmat('/home/ncj/Desktop/GMP2020/Image2pc_semantic/predicted_point/03001627pred_test_code_test.mat')
# gt = loadmat('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/03001627gt_test.mat')
pred = loadmat('/home/ncj/Desktop/GMP2020/Image2pc_semantic/predicted_point/02691156pred_test_code_val.mat')
gt = loadmat('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/02691156gt_test.mat')

snapshot_folder = '/home/ncj/Desktop/GMP2020/Image2pc_semantic/snap_shots'
synth_set = '02691156'

# pred = loadmat( '/home/ncj/Desktop/ncj/unsupervised/differentiable-point-clouds-master/pred/d2992fd5e6715bad3bbf93f83cbaf271_pc.mat' )
# gt = loadmat('/home/ncj/Desktop/ncj/unsupervised/differentiable-point-clouds-master/data/gt/downsampled/03001627/d2992fd5e6715bad3bbf93f83cbaf271.mat' )


# points = PC_Normalize(torch.from_numpy(pred['points'][0]))
print('find the data correspondence')

#for chair (格式不同)
# name_gt = ''.join(str(x) for x in gt['name']).split() # str
name_gt = gt['name'][0]
name_pred = pred['name'] # array , ''.join(pred['name'][0]) could changed to atr
n_iter = 0

for i in range (len(name_pred)):
# for i in range (5):
    #for chair
    # for j in range (len(gt['name'])):
        # if ''.join(pred['name'][i])[0:32] == gt['name'][j][0:32]:
        
    for j in range (len(gt['name'][0])):
        if ''.join(pred['name'][i])[0:32] == ''.join(gt['name'][0][j])[0:32]:
            # p1 =  torch.from_numpy(pred['points'][i]).squeeze()
            # frame = show_balls.show_3d_point_clouds(p1, False)
            # plt.imshow(frame)
            # plt.axis('off')
            # plt.savefig(snapshot_folder+'/result_image_airplane_super_val/'+str(i)+'_dpc_'+str(i) +'_point.png')
            # plt.close()
            p2 = torch.from_numpy(gt['points'][0][j]).float()
            frame = show_balls.show_3d_point_clouds(p2, False)
            print(i)
            plt.imshow(frame)
            plt.axis('off')
            plt.savefig(snapshot_folder+'/result_image_airplane_super_val/'+str(i)+'_0_pgt1' +'.png')
            plt.close()
            
            
            break
        else:
            continue



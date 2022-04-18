import os
import h5py
import numpy as np
import scipy.io as scio
from scipy.io import loadmat
import torch
from chamfer_distance import ChamferDistance
from show_pc import ShowPC
from scipy.io import loadmat
# from pc_normalize import PC_Normalize

def data_search(data, level):
    list = []
    temp = []
    for i in range(len(data)):
        if data[i] < level:
            temp.append(data[i])
        else:
            print('i: %f'%data[i])
    return [i for i in temp if i]

print('load results')
pred = loadmat('/home/ncj/Desktop/GMP2020/Image2pc_semantic/predicted_point/03001627pred_test_code_unsuper_test.mat')
gt = loadmat('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/03001627gt_test.mat')

# pred = loadmat( '/home/ncj/Desktop/ncj/unsupervised/differentiable-point-clouds-master/pred/d2992fd5e6715bad3bbf93f83cbaf271_pc.mat' )
# gt = loadmat('/home/ncj/Desktop/ncj/unsupervised/differentiable-point-clouds-master/data/gt/downsampled/03001627/d2992fd5e6715bad3bbf93f83cbaf271.mat' )


# points = PC_Normalize(torch.from_numpy(pred['points'][0]))
print('find the data correspondence')

name_gt = ''.join(str(x) for x in gt['name']).split() # str
name_pred = pred['name'] # array , ''.join(pred['name'][0]) could changed to atr
chamfer_dist = ChamferDistance()
n_iter = 0
dist1 = {}
dist2 = {}
cf = {}

# dist1[n_iter], dist2[n_iter] = chamfer_dist(torch.from_numpy(pred['points'][0]).unsqueeze(0).float(), torch.from_numpy(gt['points']).unsqueeze(0).float())
# cf[n_iter] = (torch.mean(dist1[n_iter]) + torch.mean(dist2[n_iter]))

for i in range (len(name_pred)):
# for i in range (5):
    for j in range (len(gt['name'])):
        if ''.join(pred['name'][i])[0:32] == gt['name'][j][0:32]:            
            dist1[n_iter], dist2[n_iter] = chamfer_dist(PC_Normalize(torch.from_numpy(pred['points'][i])).unsqueeze(0), PC_Normalize(torch.from_numpy(gt['points'][0][j]).float().unsqueeze(0)).unsqueeze(0))
            cf[n_iter] = torch.mean(torch.sqrt(dist1[n_iter])) + torch.mean(torch.sqrt(dist2[n_iter]))
            print ('i: %d ; cf: %f' %(i, cf[n_iter]))
            n_iter = n_iter+1
            break
        else:
            continue



ret  = data_search(cf, 1)
# len = len(cf)
# dist = sum(cf.values()) / len
Len = len(ret)
dist = sum(ret)/Len

print('OK iter = %d '% n_iter)
print('chamfer distance: %f' %dist)


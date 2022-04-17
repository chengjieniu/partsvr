import torch
import sys
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import h5py
from PIL import Image
sys.path.append("/home/ncj/Desktop/GMP2020/Image2pc_semantic/code")
from Image2pc_gan.show_pc import ShowPC
from Image2pc_gan.show_pc_seg import ShowPC_SEG
import numpy as np
class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

points_full = {}
out_dir = 'data/torch_data/'
#'Directory path to write the output.')
synth_set = '02691156'
# ganpart_data_src = h5py.File('data/torch_data/02691156partnamegan.h5')
ganpart_data_src = h5py.File('data/torch_data/02691156partgan.h5')
points_part1 = torch.from_numpy(ganpart_data_src['points_part1'][:])
points_part2 = torch.from_numpy(ganpart_data_src['points_part2'][:])
points_part3 = torch.from_numpy(ganpart_data_src['points_part3'][:])
points_part4 = torch.from_numpy(ganpart_data_src['points_part4'][:])
points_part1_name = ganpart_data_src['points_part1_name'][:]
points_part2_name = ganpart_data_src['points_part2_name'][:]
points_part3_name = ganpart_data_src['points_part3_name'][:]
points_part4_name = ganpart_data_src['points_part4_name'][:]

for i in range(0,len(points_part3)):
    print(i)
    points3 = []
    points3 = points_part3[i]
    name = points_part3_name[i]
    for j in range(0, len(points_part1)):
        points1 = []
        if name[0:20] == points_part1_name[j][0:20]:
            points1 = points_part1[j]
            break
        else:
            continue

    for j in range(0, len(points_part2)):
        points2 = []
        if name[0:20] == points_part2_name[j][0:20]:
            points2 = points_part2[j]
            break
        else:
            continue

    for j in range(0, len(points_part4)):
        points4 = points_part4[2600]
        if name[0:20] == points_part4_name[j][0:20]:
            points4 = points_part4[j]
            break
        else:
            continue
            
#chair
    # if i == 966 :
    #     continue
    # if i == 1007:
    #     continue
    # if i == 1016:
    #     continue
    # if i == 1453:
    #     continue
    # if i == 1593:
    #     continue
    # if i == 1657:
    #     continue
    # if i == 1866:
    #     continue
    # if i == 3196:
    #     continue
    # if i == 3343:
    #     continue
    # if i == 3696:
    #     continue
    # else:
    #     # ShowPC_SEG(points1, points2, points3, points4) 
    #     points_full.setdefault('points', []).append(torch.cat((points1, points2, points3, points4),0))   
#airplane
    if i == 453:
        continue
    if i == 1258:
        continue
    if i == 1395:
        continue
    if i == 1871:
        continue
    if i == 1985:
        continue
    if i == 2485:
        continue
    else:
        points_full.setdefault('points', []).append(torch.cat((points1, points2, points3, points4),0))   



#len(points_part1['name']) = 3737
#len(points_part2['name']) = 3744
#len(points_part3['name']) = 3700
#len(points_part4['name']) = 971



fh5 = h5py.File(out_dir + synth_set + 'fullgan.h5', 'w') #以'w'模式创建一个名为'test.h5'的HDF5对象
#scio.savemat(out_dir + synth_set + split_name, {"image":feature["image"], "mask":feature["mask"], "name":feature["name"]})
fh5.create_dataset('points_full', (len(points_full['points']), 8000, 3), dtype = 'f4')

# fh5.create_dataset('mask', (num_views*len(models), im_size, im_size, 1), dtype = 'f4')
# fh5.create_dataset('name', (num_views*len(models), ), dtype = h5py.special_dtype(vlen=str))
for i in range(0, len(points_full['points'])):
    fh5['points_full'][i] = np.array(points_full['points'][i], dtype = float)
fh5.close()
print("ok")
    

    











# import os
# #定义一个三维点类
# class Point(object):
# 	def __init__(self,x,y,z):
# 		self.x = x
# 		self.y = y
# 		self.z = z
# print('hello')
# points = []
# rootdir = '/media/ncj/Program/0retry_cv/shapenet_chair_pointcloud/03001627/pcd_seg_part/'
# list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
# for i in range(0,len(list)):
# 	path = os.path.join(rootdir,list[i])
# 	if os.path.isfile(path): 
# 		with open(list[i]) as f:
# 			for line in  f.readlines()[11:len(f.readlines())-1]:
# 				strs = line.split(' ')
# 				points.append(Point( strs[0], strs[1], strs[2].strip() ) )
#     ##strip()是用来去除换行符
#     ##把三维点写入txt文件

#         for i in range(len(points)):

#              linev = points[i].x+" "+points[i].y+" "+points[i].z+"\n"

#         f.close()
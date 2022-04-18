import os
import h5py
import numpy as np
import torch
from show_pc import ShowPC
import matplotlib.pyplot as plt
class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

points_full = {}


out_dir = 'data/torch_data/'
#'Directory path to write the output.')

synth_set = '03001627'
rootdir = '/media/ncj/Program/0retry_cv/shapenet_chair_pointcloud/03001627/pcd_seg_full/'

list = os.listdir(rootdir)

name = []
input_data_src = h5py.File('data/torch_data/03001627test.h5')
images = torch.from_numpy(input_data_src['image'][:])
names = input_data_src['name'][:]
masks = torch.from_numpy(input_data_src['mask'][:])
for j in range(0, int(len(names))):
    name.append( names[j]+'.pcd')

count =0
for j in range(0,len(name)):
    for i in range(0, len(list)):
        if list[i] == name[j]:
            print(j)
            path = os.path.join(rootdir, list[i])
            points = []
            if os.path.isfile(path):        
                f = open(rootdir+list[i])
                pcdfile = f.readlines()
                for line in pcdfile[11:8011]:
                    strs = line.split(' ')
                    points.append([float(strs[0]), float(strs[1]), float(strs[2].strip())]) 
            points_full.setdefault('points', []).append(points)
            points_full.setdefault('name',[]).append(list[i])
            points_full.setdefault('image', []).append(images[j])
            points_full.setdefault('mask', []).append(masks[j])
            del points  
            f.close() 


       



fh5 = h5py.File(out_dir + synth_set + 'fullgan_test.h5', 'w') #以'w'模式创建一个名为'test.h5'的HDF5对象
#scio.savemat(out_dir + synth_set + split_name, {"image":feature["image"], "mask":feature["mask"], "name":feature["name"]})
fh5.create_dataset('points', (len(points_full['points']), 8000, 3), dtype = 'f4')
fh5.create_dataset('image', (len(points_full['points']), 128, 128, 3), dtype = 'f4')
fh5.create_dataset('mask', (len(points_full['points']), 128, 128, 1), dtype = 'f4')
fh5.create_dataset('name', (len(points_full['points']), ), dtype = h5py.special_dtype(vlen=str))
for i in range(0, len(points_full['points'])):
    fh5['points'][i] = np.array(points_full['points'][i], dtype = float)
    fh5['image'][i] = np.array(points_full['image'][i], dtype = float)
    fh5['mask'][i] = np.array(points_full['mask'][i], dtype = float)
    fh5['name'][i] = np.array(points_full['name'][i], dtype = h5py.special_dtype(vlen=str))
fh5.close()
print("ok")


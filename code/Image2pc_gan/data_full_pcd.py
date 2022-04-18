import os
import h5py
import numpy as np
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
for i in range(0,len(list)):
    print(i)
    path = os.path.join(rootdir, list[i])
    points = []
    if os.path.isfile(path):        
        f = open(rootdir+list[i])
        pcdfile = f.readlines()
        for line in pcdfile[11:8011]:
            strs = line.split(' ')
            points.append([strs[0], strs[1], strs[2].strip()]) 
    points_full.setdefault('points', []).append(points)
    points_full.setdefault('name',[]).append(list[i])
    del points    

f.close()

fh5 = h5py.File(out_dir + synth_set + 'fullgan.h5', 'w') #以'w'模式创建一个名为'test.h5'的HDF5对象
#scio.savemat(out_dir + synth_set + split_name, {"image":feature["image"], "mask":feature["mask"], "name":feature["name"]})
fh5.create_dataset('points_full', (len(points_full['points']), 8000, 3), dtype = 'f4')

# fh5.create_dataset('mask', (num_views*len(models), im_size, im_size, 1), dtype = 'f4')
# fh5.create_dataset('name', (num_views*len(models), ), dtype = h5py.special_dtype(vlen=str))
for i in range(0, len(points_full['points'])):
    fh5['points_full'][i] = np.array(points_full['points'][i], dtype = float)
fh5.close()
print("ok")


import os
import h5py
import numpy as np
class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

points_part1 = {}
points_part2 = {}
points_part3 = {}
points_part4 = {}

out_dir = 'data/torch_data/'
#'Directory path to write the output.')

synth_set = '03001627'
rootdir = '/media/ncj/Program/0retry_cv/shapenet_chair_pointcloud/03001627/pcd_seg_part/'

list = os.listdir(rootdir)
for i in range(0,len(list)):
    print(i)
    path = os.path.join(rootdir, list[i])
    if list[i][-5] == '1':
        points = []
        if os.path.isfile(path):        
            f = open(rootdir+list[i])
            pcdfile = f.readlines()
            for line in pcdfile[11:2011]:
                    strs = line.split(' ')
                    points.append([strs[0], strs[1], strs[2].strip()]) 
        points_part1.setdefault('points', []).append(points)
        points_part1.setdefault('name',[]).append(list[i])
        del points
    if list[i][-5] == '2':
        points = []
        if os.path.isfile(path):        
            f = open(rootdir+list[i])
            pcdfile = f.readlines()
            for line in pcdfile[11:2011]:
                    strs = line.split(' ')
                    points.append([strs[0], strs[1], strs[2].strip()]) 
        points_part2.setdefault('points', []).append(points)
        points_part2.setdefault('name',[]).append(list[i])
        del points
    if list[i][-5] == '3':
        points = []
        if os.path.isfile(path):        
            f = open(rootdir+list[i])
            pcdfile = f.readlines()
            for line in pcdfile[11:2011]:
                    strs = line.split(' ')
                    points.append([strs[0], strs[1], strs[2].strip()]) 
        points_part3.setdefault('points', []).append(points)
        points_part3.setdefault('name',[]).append(list[i])
        del points
    if list[i][-5] == '4':
        points = []
        if os.path.isfile(path):        
            f = open(rootdir+list[i])
            pcdfile = f.readlines()
            for line in pcdfile[11:2011]:
                    strs = line.split(' ')
                    points.append([strs[0], strs[1], strs[2].strip()]) 
        points_part4.setdefault('points', []).append(points)
        points_part4.setdefault('name',[]).append(list[i])
        del points   
f.close()

#len(points_part1['name']) = 3737
#len(points_part2['name']) = 3744
#len(points_part3['name']) = 3700
#len(points_part4['name']) = 971

points = []
for j in range(0,2000):
    points.append([ 0.0, 0.0, 0.0 ])
for l in range(0,7):
    points_part1.setdefault('points', []).append(points)
    points_part1.setdefault('name',[]).append('000')
for m in range(0,44):
    points_part3.setdefault('points', []).append(points)
    points_part3.setdefault('name',[]).append('000')
for k in range(0,2773):
    points_part4.setdefault('points', []).append(points)
    points_part4.setdefault('name',[]).append('000')


fh5 = h5py.File(out_dir + synth_set + 'partgan.h5', 'w') #以'w'模式创建一个名为'test.h5'的HDF5对象
#scio.savemat(out_dir + synth_set + split_name, {"image":feature["image"], "mask":feature["mask"], "name":feature["name"]})
fh5.create_dataset('points_part1', (len(points_part1['points']), 2000, 3), dtype = 'f4')
fh5.create_dataset('points_part2', (len(points_part2['points']), 2000, 3), dtype = 'f4')
fh5.create_dataset('points_part3', (len(points_part3['points']), 2000, 3), dtype = 'f4')
fh5.create_dataset('points_part4', (len(points_part4['points']), 2000, 3), dtype = 'f4')
# fh5.create_dataset('mask', (num_views*len(models), im_size, im_size, 1), dtype = 'f4')
# fh5.create_dataset('name', (num_views*len(models), ), dtype = h5py.special_dtype(vlen=str))
for i in range(0, len(points_part1['points'])):
    fh5['points_part1'][i] = np.array(points_part1['points'][i], dtype = float)
for i in range(0, len(points_part2['points'])):
    fh5['points_part2'][i] = np.array(points_part2['points'][i], dtype = float)
for i in range(0, len(points_part3['points'])):
    fh5['points_part3'][i] = np.array(points_part3['points'][i], dtype = float)
for i in range(0, len(points_part4['points'])):
    fh5['points_part4'][i] = np.array(points_part4['points'][i], dtype = float)
fh5.close()
print('ok')
    

    











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
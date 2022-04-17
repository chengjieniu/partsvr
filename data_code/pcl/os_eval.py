import os
rootdir = '/media/ncj/Program/0retry_cv/shapenet_chair_pointcloud/02958343/obj_seg_part'
savedir = '/media/ncj/Program/0retry_cv/shapenet_chair_pointcloud/02958343/s_seg_part/'
list = os.listdir(rootdir) #list all the files and folders
print('%d',len(list))
for i in range(0,len(list)):
	path = os.path.join(rootdir,list[i])
	if os.path.isfile(path):
		print(i)
		# need 2 more parameters
		os.system("./mesh -i %s -b %s" % (rootdir+'/'+list[i], savedir+list[i][0:-4]+'.pcd'))


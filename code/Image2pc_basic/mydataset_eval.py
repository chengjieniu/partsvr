import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import h5py
from PIL import Image


class myDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        #self.dir = dir
        # input_data_src = h5py.File('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/03001627testtrain_cam.h5') #test campos fixed
        input_data_src = h5py.File('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/03001627valtrain_cam.h5') #test campos fixed
        # input_data_src = h5py.File('data/torch_data/02958343val_cam.h5')
        # input_data_src = h5py.File('data/torch_data/02691156val_cam.h5')
        images = torch.from_numpy(input_data_src['image'][:])
        masks = torch.from_numpy(input_data_src['mask'][:])
        cameras = torch.from_numpy(input_data_src['cameras'][:])
        cam_pos = torch.from_numpy(input_data_src['cam_pos'][:])
        name = input_data_src['name'][:]

        # ganfull_data_src = h5py.File('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/03001627fullgan.h5')
        # points_full = torch.from_numpy(ganfull_data_src['points_full'][:])

        # ganpart_data_src = h5py.File('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/03001627partgan.h5')
        # points_part1 = torch.from_numpy(ganpart_data_src['points_part1'][:])
        # points_part2 = torch.from_numpy(ganpart_data_src['points_part2'][:])
        # points_part3 = torch.from_numpy(ganpart_data_src['points_part3'][:])
        # points_part4 = torch.from_numpy(ganpart_data_src['points_part4'][:])

        self.images = images.float()
        self.masks = masks.float()
        self.cameras = cameras.float()
        self.cam_pos = cam_pos.float()
        self.name = name

        # self.quat =quat.float()
        # self.points = points.float()
       
        # self.points_full = points_full.float()
        # self.points_part1 = points_part1.float()
        # self.points_part2 = points_part2.float()
        # self.points_part3 = points_part3.float()
        # self.points_part4 = points_part4.float()

#test
        # input_data_src_test = h5py.File('/home/ncj/Desktop/ncj/Image2pc/code/data/torch_data/03001627testtrain_cam.h5')
        # images_test = torch.from_numpy(input_data_src_test['image'][:])
        # masks_test = torch.from_numpy(input_data_src_test['mask'][:])
        # cameras_test = torch.from_numpy(input_data_src_test['cameras'][:])
        # cam_pos_test = torch.from_numpy(input_data_src_test['cam_pos'][:])
        # self.images_test = images_test.float()
        # self.masks_test = masks_test.float()
        # self.cameras_test = cameras_test.float()
        # self.cam_pos_test = cam_pos_test.float()

    def __getitem__(self, index):
        image = self.images[index,:,:,:,:]
        mask = self.masks[index, :,:,:,:]  
        cameras = self.cameras[index, :,:, :]
        cam_pos = self.cam_pos[index, :,:]
        name = self.name[index]


          
        # return image, mask, points_part1, points_part2, points_part3, points_part4, points_full
        return image,mask,cameras, cam_pos, name
        # return image,mask,cameras, cam_pos, points_part1, points_part2, points_part3, points_part4

    def __len__(self):
     
       return len(self.images)
    #     if len(self.images)>=4704:
    #        return 1024
    #     else:
    #         return len(self.images)

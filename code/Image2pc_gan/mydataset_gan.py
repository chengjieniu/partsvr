import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import h5py
from PIL import Image


class myDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        #self.dir = dir
        # input_data_src = h5py.File('data/torch_data/03001627testtrain_cam.h5')
        # input_data_src = h5py.File('data/torch_data/03001627traintrain_cam.h5')
        input_data_src = h5py.File('data/torch_data/02691156train_cam.h5') #test campos fixed
        # input_data_src = h5py.File('data/torch_data/02958343train_cam.h5')
        images = torch.from_numpy(input_data_src['image'][:])
        masks = torch.from_numpy(input_data_src['mask'][:])
        cameras = torch.from_numpy(input_data_src['cameras'][:])
        cam_pos = torch.from_numpy(input_data_src['cam_pos'][:])

        # ganfull_data_src = h5py.File('data/torch_data/03001627fullgan.h5')
        ganfull_data_src = h5py.File('data/torch_data/02691156fullgan.h5')
        # ganfull_data_src = h5py.File('data/torch_data/02958343fullgan.h5')
        points_full = torch.from_numpy(ganfull_data_src['points_full'][:])

        # ganpart_data_src = h5py.File('data/torch_data/03001627partgan.h5')
        ganpart_data_src = h5py.File('data/torch_data/02691156partgan.h5')
        # ganpart_data_src = h5py.File('data/torch_data/02958343partgan.h5')
        points_part1 = torch.from_numpy(ganpart_data_src['points_part1'][:])
        points_part2 = torch.from_numpy(ganpart_data_src['points_part2'][:])
        points_part3 = torch.from_numpy(ganpart_data_src['points_part3'][:])
        points_part4 = torch.from_numpy(ganpart_data_src['points_part4'][:])

        self.images = images.float()
        self.masks = masks.float()
        self.cameras = cameras.float()
        self.cam_pos = cam_pos.float()

        # self.quat =quat.float()
        # self.points = points.float()
       
        self.points_full = points_full.float()
        self.points_part1 = points_part1.float()
        self.points_part2 = points_part2.float()
        self.points_part3 = points_part3.float()
        self.points_part4 = points_part4.float()

    def __getitem__(self, index):
        image = self.images[index%512,:,:,:,:]
        mask = self.masks[index%512,:,:,:,:]  
        cameras = self.cameras[index%512, :,:, :]
        cam_pos = self.cam_pos[index%512, :,:]

        # quat = self.quat[index, :]
        # points =self.points[index, :,:]

        points_full = self.points_full[index%self.points_full.shape[0],:,:]
        points_part1 = self.points_part1[index%self.points_part1.shape[0],:,:]
        points_part2 = self.points_part2[index%self.points_part2.shape[0],:,:]
        points_part3 = self.points_part3[index%self.points_part3.shape[0],:,:]
        points_part4 = self.points_part4[index%self.points_part4.shape[0],:,:]
          
        # return image, mask, points_part1, points_part2, points_part3, points_part4, points_full
        return image,mask,cameras, cam_pos, points_part1, points_part2, points_part3, points_part4, points_full
        # return image,mask,cameras, cam_pos, points_part1, points_part2, points_part3, points_part4

    def __len__(self):
     
       # return len(self.images)
        if len(self.images)>4704:
           return 512

        # else:
        #     return len(self.images)
        if len(self.images)>2828:
           return 2828

        # if len(self.images) > 5244:
        #     return 1024

        else:
            return len(self.images)
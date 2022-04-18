import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import h5py
from PIL import Image


class myDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        #self.dir = dir
        # input_data_src = h5py.File('data/torch_data/03001627train.h5')
        # input_data_src = h5py.File('/home/dell/ncj/image_pc_code/data/torch_data/03001627traintrain_cam.h5') #test campos fixed

        input_data_src = h5py.File( 'data/torch_data/03001627traintrain_cam.h5' )
        images = torch.from_numpy(input_data_src['image'][:])
        masks = torch.from_numpy(input_data_src['mask'][:])
        cameras = torch.from_numpy(input_data_src['cameras'][:])
        cam_pos = torch.from_numpy(input_data_src['cam_pos'][:])

        #quat = torch.from_numpy(input_data_src['campos'][:])
        # points = torch.from_numpy(input_data_src['points'][:])


        # ganfull_data_src = h5py.File('data/torch_data/03001627fullgan.h5')
        # points_full = torch.from_numpy(ganfull_data_src['points_full'][:])

        # ganpart_data_src = h5py.File('data/torch_data/03001627partgan.h5')
        # points_part1 = torch.from_numpy(ganpart_data_src['points_part1'][:])
        # points_part2 = torch.from_numpy(ganpart_data_src['points_part2'][:])
        # points_part3 = torch.from_numpy(ganpart_data_src['points_part3'][:])
        # points_part4 = torch.from_numpy(ganpart_data_src['points_part4'][:])

        self.images = images.float()
        self.masks = masks.float()
        self.cameras = cameras.float()
        self.cam_pos = cam_pos.float()

        # self.quat =quat.float()
        # self.points = points.float()

        # self.points_full = points_full.float()
        # self.points_part1 = points_part1.float()
        # self.points_part2 = points_part2.float()
        # self.points_part3 = points_part3.float()
        # self.points_part4 = points_part4.float()

    def __getitem__(self, index):
        image = self.images[index%128,:,:,:,:]
        mask = self.masks[index%128,:,:,:,:]
        cameras = self.cameras[index%128, :,:, :]
        cam_pos = self.cam_pos[index%128, :,:]

        # quat = self.quat[index, :]
        # points =self.points[index, :,:]

        # points_full = self.points_full[index%self.points_full.shape[0],:,:]
        # points_part1 = self.points_part1[index%self.points_part1.shape[0],:,:]
        # points_part2 = self.points_part2[index%self.points_part2.shape[0],:,:]
        # points_part3 = self.points_part3[index%self.points_part3.shape[0],:,:]
        # points_part4 = self.points_part4[index%self.points_part4.shape[0],:,:]
          
        # return image, mask, points_part1, points_part2, points_part3, points_part4, points_full
        return image,mask,cameras, cam_pos
        # return image, mask

    def __len__(self):
        if len(self.images) > 1024:
            return 128
        else:
            return len(self.images)

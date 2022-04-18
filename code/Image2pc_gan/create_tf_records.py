import sys
import os
import glob
import re
import random

import numpy as np
import scipy.io as scio
from scipy.io import loadmat
from imageio import imread

from skimage.transform import resize as im_resize

from util.fs import mkdir_if_missing
from util.data import tf_record_options
import matplotlib.pyplot as plt
import h5py, os

import tensorflow as tf

from tensorflow import app

split_dir =  'data/splits/'        
#'Directory path containing the input rendered images.'

inp_dir_renders = 'data/renders/'
#'Directory path containing the input rendered images.'

inp_dir_voxels = ''
# 'Directory path containing the input voxels.'

out_dir = 'data/torch_data/'
#'Directory path to write the output.')

synth_set = '03001627'

store_camera = True
store_voxels = False
store_depth = False
split_path = '', ''

num_views = 5
#'Num of viewpoints in the input data.')
image_size = 128
#'Input images dimension (pixels) - width & height.')
vox_size= 64
#'Voxel prediction dimension.')
tfrecords_gzip_compressed = True 
#'Voxel prediction dimension.')



def read_camera(filename):
    cam = loadmat(filename)
    extr = cam["extrinsic"]
    pos = cam["pos"]
    quat = cam["quat"]
    return extr, pos,quat


def loadDepth(dFile, minVal=0, maxVal=10):
    dMap = imread(dFile)
    dMap = dMap.astype(np.float32)
    dMap = dMap*(maxVal-minVal)/(pow(2,16)-1) + minVal
    return dMap


# def _dtype_feature(ndarray):
#     ndarray = ndarray.flatten()
#     """match appropriate tf.train.Feature class with dtype of ndarray. """
#     assert isinstance(ndarray, np.ndarray)
#     dtype_ = ndarray.dtype
#     if dtype_ == np.float64 or dtype_ == np.float32:
#         return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray))
#     elif dtype_ == np.int64:
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=ndarray))
#     else:
#         raise ValueError("The input should be numpy ndarray. \
#                            Instaed got {}".format(ndarray.dtype))


def _string_feature(s):
    s = s.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s]))


def create_record(synth_set, split_name, models):
    im_size = image_size
    num_views = 5
    num_models = len(models)

    mkdir_if_missing(out_dir)

    # address to save the TFRecords file
    train_filename = "{}/{}_{}.tfrecords".format(out_dir, synth_set, split_name)
    # open the TFRecords file
    # options = tf_record_options()
    # writer = tf.python_io.TFRecordWriter(train_filename, options=options)

    render_dir = os.path.join(inp_dir_renders, synth_set)
    voxel_dir = os.path.join(inp_dir_voxels, synth_set)
    rgbs = np.zeros((len(models), num_views, im_size, im_size, 3), dtype=np.float32)
    masks = np.zeros((len(models), num_views, im_size, im_size, 1), dtype=np.float32)
    cam_quat = np.zeros((len(models), num_views, 4), dtype=np.float32)
    cameras = np.zeros((len(models), num_views, 4, 4), dtype=np.float32)
    cam_pos = np.zeros((len(models), num_views, 3), dtype=np.float32)
    for j, model in enumerate(models):
        print("{}/{}".format(j, num_models))

        if store_voxels:
            voxels_file = os.path.join(voxel_dir, "{}.mat".format(model))
            voxels = loadmat(voxels_file)["Volume"].astype(np.float32)

            # this needed to be compatible with the
            # PTN projections
            voxels = np.transpose(voxels, (1, 0, 2))
            voxels = np.flip(voxels, axis=1)

        im_dir = os.path.join(render_dir, model)
        images = sorted(glob.glob("{}/render_*.png".format(im_dir)))

        
      
        
        depths = np.zeros((num_views, im_size, im_size, 1), dtype=np.float32)

        assert(len(images) >= num_views)

        for k in range(num_views):
            im_file = images[k]
            img = imread(im_file)
            rgb = img[:, :, 0:3]
            mask = img[:, :, [3]]
            mask = mask / 255.0
            if True:  # white background
                mask_fg = np.repeat(mask, 3, 2)
                mask_bg = 1.0 - mask_fg
                rgb = rgb * mask_fg + np.ones(rgb.shape)*255.0*mask_bg
            # plt.imshow(rgb.astype(np.uint8))
            # plt.show()
            rgb = rgb / 255.0
            actual_size = rgb.shape[0]
            if im_size != actual_size:
                rgb = im_resize(rgb, (im_size, im_size), order=3)
                mask = im_resize(mask, (im_size, im_size), order=3)
            rgbs[j, k, :, :, :] = rgb
            masks[j, k, :, :, :] = mask

            fn = os.path.basename(im_file)
            img_idx = int(re.search(r'\d+', fn).group())

            if store_camera:
                cam_file = "{}/camera_{}.mat".format(im_dir, img_idx)
                cam_extr, pos, quat = read_camera(cam_file)
                cameras[j, k, :, :] = cam_extr
                cam_pos[j, k, :] = pos
                cam_quat[j, k, :] = quat


            if store_depth:
                depth_file = "{}/depth_{}.png".format(im_dir, img_idx)
                depth = loadDepth(depth_file)
                d_max = 10.0
                d_min = 0.0
                depth = (depth - d_min) / d_max
                depth_r = im_resize(depth, (im_size, im_size), order=0)
                depth_r = depth_r * d_max + d_min
                depths[k, :, :] = np.expand_dims(depth_r, -1)

    # Create a feature
    # feature = {"image": rgbs,
    #         "mask": masks,
    #         "name": models}
    f = h5py.File(out_dir + synth_set + split_name + 'train_cam.h5', 'w') #以'w'模式创建一个名为'test.h5'的HDF5对象
    #scio.savemat(out_dir + synth_set + split_name, {"image":feature["image"], "mask":feature["mask"], "name":feature["name"]})
    f.create_dataset('image', (len(models), num_views, im_size, im_size, 3), dtype = 'f4')
    f.create_dataset('mask', (len(models), num_views, im_size, im_size, 1), dtype = 'f4')
    f.create_dataset('name', (num_views*len(models), ), dtype = h5py.special_dtype(vlen=str))
    f.create_dataset('campos', (len(models), num_views, 4), dtype= 'f4')
    f.create_dataset('cameras',(len(models), num_views, 4,4), dtype='f4')
    f.create_dataset('cam_pos', (len(models), num_views, 3), dtype='f4')

    # f.create_dataset('voxels', (num_views, 64, 64, 64), dtype= 'f4')
    for i in range(len(models)):
        f['image'][i] = rgbs[i]
        f['mask'][i]= masks[i]
        f['name'][i] = models[int(i)]
        f['campos'][i] = cam_quat[i]
        f['cameras'][i] = cameras[i]
        f['cam_pos'][i]= cam_pos[i]
        # f['voxels'][i] = voxels[i]
        print(i)

    f.close()
    return None

        # if store_voxels:
        #     feature["vox"] = voxels

        # if store_camera:
        #     # feature["extrinsic"] = _dtype_feature(extrinsic)
        #     feature["extrinsic"] = cameras
        #     feature["cam_pos"] = cam_pos

        # if store_depth:
        #     feature["depth"] = depths

        # # Create an example protocol buffer
        # example = tf.train.Example(features=tf.train.Features(feature=feature))
        # # Serialize to string and write on the file
        # writer.write(example.SerializeToString())

        # """
        # plt.imshow(np.squeeze(img[:,:,0:3]))
        # plt.show()
        # plt.imshow(np.squeeze(img[:,:,3]).astype(np.float32)/255.0)
        # plt.show()
        # """        

    





SPLIT_DEF = [("val", 0.05), ("train", 0.95)]


def generate_splits(input_dir):
    files = [f for f in os.listdir(input_dir) if os.path.isdir(f)]
    models = sorted(files)
    random.shuffle(models)
    num_models = len(models)
    models = np.array(models)
    out = {}
    first_idx = 0
    for k, splt in enumerate(SPLIT_DEF):
        fraction = splt[1]
        num_in_split = int(np.floor(fraction * num_models))
        end_idx = first_idx + num_in_split
        if k == len(SPLIT_DEF)-1:
            end_idx = num_models
        models_split = models[first_idx:end_idx]
        out[splt[0]] = models_split
        first_idx = end_idx
    return out


def load_drc_split(base_dir, synth_set):
    filename = os.path.join(base_dir, "{}.file".format(synth_set))
    lines = [line.rstrip('\n') for line in open(filename)]

    k = 3  # first 3 are garbage
    split = {}
    while k < len(lines):
        _,_,name,_,_,num = lines[k:k+6]
        k += 6
        num = int(num)
        split_curr = []
        for i in range(num):
            _, _, _, _, model_name = lines[k:k+5]
            k += 5
            split_curr.append(model_name)
        split[name] = split_curr

    return split


def generate_records(synth_set):
    base_dir = split_dir
    split = load_drc_split(base_dir, synth_set)

    for key, value in split.items():
        create_record(synth_set, key, value)


def read_split(filename):
    f = open(filename, "r")
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def main():
    generate_records(synth_set)


if __name__ == '__main__':
    main()

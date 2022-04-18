import numpy as np
import torch 
import math
import time
from torch.autograd import Variable
from torch import nn
import util.drc
from util.quaternion import quaternion_rotate
from util.camera import intrinsic_matrix
from util.point_cloud_distance import *
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
from skimage import io,data
from show_pc import ShowPC
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from time import strftime, gmtime
from pose_decoder import PoseDecoder
# from pc_decoder import PCDecoder
from model_pc import get_dropout_prob
from torch.nn.functional import pad

def multi_expand(inp, axis, num):
    inp_big = inp
    for i in range(num):
        inp_big = torch.unsqueeze(inp_big, axis)
    return inp_big

def pointcloud2voxels(input_pc, sigma):
    x = input_pc[:, :, 0] # [B,N,3]
    y = input_pc[:, :, 1]
    z = input_pc[:, :, 2]

    vox_size = 64

    rng = torch.linspace(-1.0, 1.0, vox_size)
    xg, yg, zg = torch.meshgrid(rng, rng, rng) # [G,G,G]

    x_big = multi_expand(x, -1, 3) # [B,N,1,1,1]
    y_big = multi_expand(y, -1, 3) # [B,N,1,1,1]
    z_big = multi_expand(z, -1, 3) # [B,N,1,1,1]

    xg = multi_expand(xg, 0, 2) # [1,1,G,G,G]
    yg = multi_expand(yg, 0, 2) # [1,1,G,G,G]
    zg = multi_expand(zg, 0, 2) # [1,1,G,G,G]

    #squared distance
    sq_distance = torch.pow(x_big - xg, 2) + torch.pow(y_big - yg, 2) + torch.pow(z_big - zg, 2)

    #compute gaussian
    func = torch.exp(-sq_distance / (2.0 * sigma * sigma)) # [B,N,G,G,G]

    #normal guassion
    # should work with any grid sizes
    magic_facotr = 1.78984352254  # see estimate_gauss_normaliser
    sigma_normalised = sigma * vox_size
    normaliser = 1.0 / (magic_facotr * torch.pow(sigma_normalised, 3))
    func *= normaliser

    summed = torch.sum(func, axis=1) # [B,G,G G]
    voxels = torch.clamp(summed, 0.0, 1.0)
    voxels = torch.unsqueeze(voxels, axis=-1)  # [B,G,G,G,1]

    return voxels

def pc_perspective_transform(point_cloud, transform, predicted_translation=None, focal_length=None):
    """
    :param cfg:
    :param point_cloud: [B, N, 3]
    :param transform: [B, 4] if quaternion or [B, 4, 4] if camera matrix
    :param predicted_translation: [B, 3] translation vector
    :return:
    """
    camera_distance = 2.0
    if focal_length is None:
        focal_length = 1.875
    else:
        focal_length = torch.unsqueeze(focal_length, dim = -1)

    # Represent camera rotation as quaternion instead of matrix.
    pc2 = quaternion_rotate(point_cloud, transform)

    indices_x = torch.LongTensor([2])
    xs = torch.index_select(pc2, 2, indices_x.cuda())
    indices_y = torch.LongTensor([1])
    ys = torch.index_select(pc2, 2, indices_y.cuda())
    indices_z = torch.LongTensor([0])
    zs = torch.index_select(pc2, 2, indices_z.cuda())

    zs = zs + camera_distance
    xs = xs * focal_length
    ys = ys * focal_length

    xs = xs / zs
    ys = ys / zs

    zs = zs - camera_distance
    xyz2 = torch.cat([zs, ys, xs], dim = 2)
    return xyz2

def pointcloud2voxels3d_fast( pc):  # [B,N,3]
    vox_size = 64
    
    vox_size_z = vox_size

    batch_size = pc.shape[0]
    num_points = pc.shape[1]

    has_rgb = False 

    grid_size = 1.0
    half_size = grid_size / 2
    filter_outliers = True
    valid = (pc >= -half_size) & (pc <= half_size)
    valid = valid[:,:,0]&valid[:,:,1]&valid[:,:,2]


    vox_size_tf = torch.Tensor([[[ vox_size_z, vox_size, vox_size]]])
    pc_grid = (pc + half_size) * (vox_size_tf.cuda() - 1)
    indices_floor = torch.floor(pc_grid)
    indices_int = indices_floor.int()
    batch_indices = torch.arange(0, batch_size, 1).cuda()
    batch_indices = torch.unsqueeze(batch_indices, -1)
    batch_indices = batch_indices.repeat([1, num_points])
    batch_indices = torch.unsqueeze(batch_indices, -1).int()

    indices = torch.cat([batch_indices, indices_int], dim=2)
    indices = indices.reshape([-1, 4])
    # print(torch.max(indices))

    r = pc_grid - indices_floor  # fractional part
    rr = [1.0 - r, r]
    
    if filter_outliers:
        valid = torch.reshape(valid, [-1])
        valid = torch.unsqueeze(valid, -1)
        indices = torch.masked_select(indices, valid.byte())
        indices = indices.reshape([-1, 4])
        

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][:, :, 0] * rr[pos[1]][:, :, 1] * rr[pos[2]][:, :, 2]
        updates = torch.reshape(updates_raw, [-1])
        updates = torch.unsqueeze(updates, -1)
        if filter_outliers:
            updates = torch.masked_select(updates, valid.byte())
            try:
                valid.unique()
            except:
                print("Exception has occurred: RuntimeError\
                    unique: failed to synchronize: device-side assert triggered")           
  
           
        indices_loc = indices
        indices_shift = torch.Tensor([[0] + pos]).int().cuda()
        num_updates = indices_loc.shape[0]
        indices_shift = indices_shift.repeat([num_updates, 1])
        indices_loc = (indices_loc + indices_shift).long()     
        
        voxels = torch.zeros([batch_size, vox_size_z, vox_size, vox_size]).cuda()
        # print(torch.max(indices_loc))
        indices_loc[indices_loc>63]=63
        indices_loc[indices_loc<0]=0
        voxels[indices_loc[:,0], indices_loc[:,1], indices_loc[:,2], indices_loc[:,3]] = updates

        try:
            valid.unique()
        except:
            print("Exception has occurred: RuntimeError\
                unique: failed to synchronize: device-side assert triggered")

        if has_rgb:
            if cfg.pc_rgb_stop_points_gradient:
                updates_raw = torch.stop_gradient(updates_raw)
            updates_rgb = torch.unsqeeze(updates_raw, axis=-1) * rgb
            updates_rgb = torch.reshape(updates_rgb, [-1, 3])
            if filter_outliers:
                updates_rgb = torch.masked_select(updates_rgb, valid.byte())
            voxels_rgb = torch.scatter_(indices_loc, updates_rgb, [batch_size, vox_size_z, vox_size, vox_size, 3])
        else:
            voxels_rgb = None

        return voxels, voxels_rgb

    
    voxels = []
    voxels_rgb = []
    # start = time.time()
    for k in range(2):
        for j in range(2):
            for i in range(2):       
               
                vx, vx_rgb = interpolate_scatter3d([k, j, i])
                voxels.append(vx)
                voxels_rgb.append(vx_rgb)
    # print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    voxels = voxels[0] + voxels[1] +voxels[2] +voxels[3] +voxels[4] + voxels[5] + voxels[6] +voxels[7]
    voxels_rgb = voxels_rgb[0] + voxels_rgb[1] +voxels_rgb[2] +voxels_rgb[3] +voxels_rgb[4] + voxels_rgb[5] + voxels_rgb[6] +voxels_rgb[7] if has_rgb else None

    return voxels, voxels_rgb

def pointcloud_project(point_cloud, transform, sigma):
    tr_pc = pc_perspective_transform(point_cloud, transform)
    #tr_pc = tr_pc.cpu()
    voxels = pointcloud2voxels3d_fast(tr_pc)
    voxels = voxels.permute(0,2,1,3,4)

    proj, probs = util.drc.drc_projection(voxels)
    idx = [i for i in range(proj.size(1)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    proj = torch.index_select(proj, 1, idx)
    return proj, voxels

def conv3d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    batch_size, channel, height, rows, cols = input.size()
    out_c, input_c, filter_height, filter_rows, filter_cols  = weight.size()
    effective_filter_size_rows = (filter_rows - 1)  + 1
    effective_filter_size_cols = (filter_cols - 1)  + 1
    effective_filter_size_height = (filter_height - 1)  + 1
    out_rows = (rows + stride - 1) // stride
    out_cols = (cols + stride - 1) // stride
    out_height = (height + stride - 1) // stride
    padding_rows = max(0, (out_rows - 1) * stride +
                        effective_filter_size_rows - rows)
    padding_cols = max(0, (out_cols - 1) * stride +
                        effective_filter_size_cols - cols)
    padding_height = max(0, (out_height - 1) * stride +
                        effective_filter_size_height - height)

    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_front = int(padding_height / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    padding_back = padding_height - padding_front
    paddings = [padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back]
    # input = torch.nn.ZeroPad3d(paddings)(input)
    input = pad(input,paddings)
    return F.conv3d(input, weight, bias, 1,)


def smoothen_voxels3d(voxels, kernel):
    # if cfg.pc_separable_gauss_filter:
    if True:
        for krnl in kernel:
            voxels = conv3d_same_padding(voxels, krnl.permute(4,3,0,1,2).cuda(), None, stride=1)
    else:
        voxels = nn.functional.conv3d(voxels, kernel.permute(4,3,0,1,2).cuda(), padding = 1)
    return voxels

def pointcloud_project_fast(cfg,  point_cloud, transform, kernel = None, scaling_factor=None, focal_length=None):
    tr_pc = pc_perspective_transform(point_cloud, transform, None, focal_length)

    voxels, voxels_rgb= pointcloud2voxels3d_fast(tr_pc)
    voxels = torch.unsqueeze(voxels, dim=-1)

    voxels_raw = voxels

    voxels = torch.clamp(voxels, min = 0.0, max = 1.0)

    if kernel is not None:
        voxels = voxels.permute(0,4,1,2,3)
        voxels = smoothen_voxels3d(voxels, kernel)
        voxels = voxels.permute(0,2,3,4,1)
        # if has_rgb:
        #     if not cfg.pc_rgb_clip_after_conv:
        #         voxels_rgb = torch.clamp(voxels_rgb, min = 0.0, max = 1.0)
        #     voxels_rgb = convolve_rgb(cfg, voxels_rgb, kernel)

    if scaling_factor is not None:
        sz = scaling_factor.shape[0]
        scaling_factor = torch.reshape(scaling_factor, [sz, 1, 1, 1, 1])
        voxels = voxels * scaling_factor
        voxels = torch.clamp(voxels, min = 0.0, max = 1.0)

    proj, probs = util.drc.drc_projection(voxels, cfg)

    idx = [i for i in range(proj.size(1)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    proj = torch.index_select(proj, 1, idx.cuda())
    # proj = proj.flip(0,2,1,3)
    # proj = proj.permute(0,2,1,3)

    return proj, voxels

  
def get_dropout_keep_prob(cfg, params, iter):
    return get_dropout_prob(cfg, params['_global_step'], iter)

def pc_point_dropout(points, rgb, keep_prob):
    shape = points.shape
    num_input_points = shape[1]
    batch_size = shape[0]
    num_channels = shape[2]
    num_output_points = int(num_input_points * keep_prob)
    # print(num_output_points)

    def sampler(num_output_points_np):
        all_inds = []
        for k in range(batch_size):
            ind = np.random.choice(num_input_points, num_output_points_np, replace=False)
            ind = np.expand_dims(ind, axis=-1)
            ks = np.ones_like(ind) * k
            inds = np.concatenate((ks, ind), axis=1)
            all_inds.append(np.expand_dims(inds, 0))
        return np.concatenate(tuple(all_inds), 0).astype(np.int64)

    selected_indices = sampler(num_output_points)
    idx1, idx2 = torch.tensor(selected_indices).chunk(2,dim = -1)
    out_points = points[idx1,idx2]
    out_points = torch.reshape(out_points, [batch_size, num_output_points, num_channels])
    if rgb is not None:
        num_rgb_channels = rgb.shape.as_list()[2]
        out_rgb = tf.gather_nd(rgb, selected_indices)
        out_rgb = tf.reshape(out_rgb, [batch_size, num_output_points, num_rgb_channels])
    else:
        out_rgb = None
    return out_points, out_rgb

def compute_projection (inputs, outputs, is_training, cfg, params, iter):
    proj_out = dict()
    all_points = outputs['all_points']
    all_rgb = outputs['all_rgb']

    if cfg.predict_pose:
        camera_pose = outputs['poses']
    else:
        if cfg.pose_quaternion:
            camera_pose = inputs['camera_quaternion']
        else:
            camera_pose = inputs['matrices']
    if is_training and cfg.pc_point_dropout != 1:
        dropout_prob = get_dropout_keep_prob(cfg, params, iter)
        all_points, all_rgb = pc_point_dropout(all_points, all_rgb, dropout_prob)
    
    if is_training  is False:
        proj, voxels = pointcloud_project_fast(cfg, all_points, camera_pose, None, 
                                            scaling_factor = outputs['all_scaling_factors'],
                                            focal_length = outputs['focal_length'])
    else: 
        proj, voxels = pointcloud_project_fast(cfg, all_points, camera_pose, params['_gauss_kernel'], 
                                            scaling_factor = outputs['all_scaling_factors'],
                                            focal_length = outputs['focal_length'])
    # outputs["projs_rgb"] = None
    # outputs["projs_depth"] = None
    proj_out['projs'] = proj

    batch_size = outputs['points_1'].shape[0]
    proj_out['projs_1'] = proj[0:batch_size, :, :, :]

    return proj_out

# if __name__ == "__main__":
#     outputs = dict()
#     point_cloud = torch.randn((1,8000,3)).cuda()
#     transform = torch.randn(1,4).cuda()
#     outputs['points_1'] = Variable(point_cloud, requires_grad = True)
#     outputs['poses'] = Variable(transform, requires_grad =True)
#     out = compute_projection(outputs)
#     x = torch.zeros(1, 64,64,1)
#     loss_fn = torch.nn.MSELoss(reduce = True, size_average =True).cuda()
#     loss = loss_fn(x.cuda(), out['projs_1'])
#     loss.backward()
#     print(loss)

# if __name__ == "__main__":
#     decoder_pose = PoseDecoder()
#     decoder_pc = PCDecoder()

#     decoder_pose_opt = torch.optim.Adam(decoder_pose.parameters(), lr = 1e-2,weight_decay = 1e-1)
#     decoder_pose.cuda()
#     decoder_pc_opt = torch.optim.Adam(decoder_pc.parameters(), lr = 1e-5, weight_decay = 1e-6)
#     decoder_pc.cuda()

#     outputs = dict()
#     # points = loadmat("points.mat") 
#     # points = points["points"][0:8000]
#     points = torch.randn(8000, 3).cuda()
#     points = torch.Tensor(points)
#     points = points.unsqueeze(0).cuda()
#     transform = loadmat("camera_0.mat")
#     transform = torch.Tensor(transform["quat"]).cuda()
#     # transform = torch.randn(1,4).cuda()

#     outputs['points_1'] = points
#     # outputs['poses'] = Variable(transform, requires_grad = True)
#     outputs['poses'] = transform
#     IN_Mask = torch.tensor(loadmat('ms.mat')['ms'])
#     IN_Mask = IN_Mask.unsqueeze(0).unsqueeze(-1)
#     IN_Mask = IN_Mask.permute(0,3,1,2)
#     IN_Mask = F.interpolate(IN_Mask,[64,64])
#     IN_Mask = IN_Mask.permute(0,2,3,1)
    
#     start= time.time()
#     for i in range(0,50000):
#         decoder_pose_opt.zero_grad()
#         decoder_pc_opt.zero_grad()
#         dec_pose = decoder_pose(outputs['poses'])
#         outputs.update(dec_pose)


#         dec_pc  = decoder_pc(outputs['points_1'])
#         outputs.update(dec_pc)

#         outputs['all_scaling_factors'] =None
#         outputs['focal_length'] =None

#         out = compute_projection(outputs)
#         task_loss = torch.nn.MSELoss().cuda()
#         loss = task_loss(IN_Mask,out['projs'].cpu())
#         loss.backward(retain_graph=True)
#         decoder_pose_opt.step()
#         decoder_pc_opt.step()
#         print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)), ' iteration: %d, loss: %f' %(i,  loss))
    
#     with SummaryWriter(comment='compute_projection') as w:
#         w.add_graph(out, (outputs,))
#     print("ok")



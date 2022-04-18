#import tensorflow as tf
import torch
import math


"""
def drc_projection2(transformed_voxels):
    # swap batch and Z dimensions for the ease of processing
    input = tf.transpose(transformed_voxels, [1, 0, 2, 3, 4])

    y = input
    x = 1.0 - y

    v_shape = tf.shape(input)
    size = v_shape[0]
    print("num z", size)

    # this part computes tensor of the form [1, x1, x1*x2, x1*x2*x3, ...]
    init = tf.TensorArray(dtype=tf.float32, size=size)
    init = init.write(0, slice_axis0(x, 0))
    index = (1, x, init)

    def cond(i, _1, _2):
        return i < size

    def body(i, input, accum):
        prev = accum.read(i)
        print("previous val", i, prev.shape)
        new_entry = prev * input[i, :, :, :, :]
        new_i = i + 1
        return new_i, input, accum.write(new_i, new_entry)

    r = tf.while_loop(cond, body, index)[2]
    outp = r.stack()

    out = tf.reduce_max(transformed_voxels, [1])
    return out, outp
"""


DTYPE = torch.FloatTensor


def slice_axis0(t, idx):
    init = t[idx, :, :, :, :]
    return torch.unsqueeze(init, dim=0)


def drc_event_probabilities_impl(voxels, cfg):
    # swap batch and Z dimensions for the ease of processing
    input = voxels.permute(1, 0, 2, 3, 4)
    logsum = cfg.drc_logsum
    dtp = DTYPE

    clip_val = cfg.drc_logsum_clip_val
    if logsum:
        input = torch.clamp(input, clip_val, 1.0-clip_val)

    def log_unity(shape):
        return torch.ones(shape)*clip_val

    y = input
    x = 1.0 - y
    if logsum:
        y = torch.log(y)
        x = torch.log(x)
        op_fn = torch.add
        unity_fn = log_unity
        cum_fun = torch.cumsum
    else:
        op_fn = torch.mul
        unity_fn = torch.ones
        cum_fun = torch.cumprod

    v_shape = input.shape
    singleton_shape = [1, v_shape[1], v_shape[2], v_shape[3], v_shape[4]]

    # this part computes tensor of the form,
    # ex. for vox_size=3 [1, x1, x1*x2, x1*x2*x3]
    if cfg.drc_tf_cumulative:       
        r = cum_fun(x, dim=0)

    r1 = unity_fn(singleton_shape)
    p1 = torch.cat([r1.cuda(), r], dim=0)  # [1, x1, x1*x2, x1*x2*x3]

    r2 = unity_fn(singleton_shape)
    p2 = torch.cat([y, r2.cuda()], dim=0)  # [(1-x1), (1-x2), (1-x3), 1])

    p = op_fn(p1, p2)  # [(1-x1), x1*(1-x2), x1*x2*(1-x3), x1*x2*x3]
    if logsum:
        p = torch.exp(p)

    return p, singleton_shape, input


def drc_event_probabilities(voxels):
    p, _, _ = drc_event_probabilities_impl(voxels)
    return p


def drc_projection(voxels, cfg):
    p, singleton_shape, input = drc_event_probabilities_impl(voxels, cfg)
    dtp = DTYPE

    # colors per voxel (will be predicted later)
    # for silhouettes simply: [1, 1, 1, 0]
    c0 = torch.ones_like(input)
    c1 = torch.zeros(singleton_shape).cuda()
    c = torch.cat([c0, c1], dim=0)

    # \sum_{i=1:vox_size} {p_i * c_i}
    out = torch.sum(p * c, [0])

    return out, p


def project_volume_rgb_integral( p, rgb):
    # swap batch and z
    rgb = rgb.permute(1, 0, 2, 3, 4)
    v_shape = rgb.shape
    singleton_shape = [1, v_shape[1], v_shape[2], v_shape[3], v_shape[4]]
    background = torch.ones(shape=singleton_shape)
    rgb_full = torch.cat([rgb, background], dim=0)

    out = torch.sum(p * rgb_full, [0])

    return out


def drc_depth_grid(z_size):
    i_s = torch.range(0, z_size, 1)
    di_s = i_s / z_size - 0.5 + 2.0
    last = torch.Tensor([10])
    return torch.cat([di_s, last], dim=0)


def drc_depth_projection(p):
    z_size = p.shape[0]
    z_size = z_size - 1  # because p is already of size vox_size + 1
    depth_grid = drc_depth_grid( z_size)
    psi = torch.reshape(depth_grid, shape=[-1, 1, 1, 1, 1])
    # \sum_{i=1:vox_size} {p_i * psi_i}
    out = torch.sum(p * psi, [0])
    return out

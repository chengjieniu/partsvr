import numpy as np
import scipy.io
import torch 
from torch.autograd import Variable

from model_base import pool_single_view

from util.gauss_kernel import gauss_smoothen_image, smoothing_kernel
from util.quaternion import \
    quaternion_multiply as q_mul,\
    quaternion_normalise as q_norm,\
    quaternion_rotate as q_rotate,\
    quaternion_conjugate as q_conj

def tf_repeat_0(input, num):
    orig_shape = input.shape
    e = input.unsqueeze(1)
    tiler = [1 for _ in range(len(orig_shape)+1)]
    tiler[1] = num
    tiled = e.repeat(tiler)
    new_shape = [-1]
    new_shape.extend(orig_shape[1:])
    final = tiled.reshape( new_shape)
    return final

def get_smooth_sigma(cfg, global_step, num_iter):
    num_steps = cfg.epochs*num_iter
    diff = (cfg.pc_relative_sigma_end - cfg.pc_relative_sigma)
    sigma_rel = cfg.pc_relative_sigma + global_step / num_steps * diff
    return sigma_rel


def get_dropout_prob(cfg, global_step, num_iter):
    if not cfg.pc_point_dropout_scheduled:
        return cfg.pc_point_dropout

    exp_schedule = cfg.pc_point_dropout_exponential_schedule
    num_steps = cfg.epochs*num_iter
    keep_prob_start = cfg.pc_point_dropout
    keep_prob_end = 1.0
    start_step = cfg.pc_point_dropout_start_step
    end_step = cfg.pc_point_dropout_end_step
    global_step = float(global_step)
    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    if exp_schedule:
        alpha = torch.log(keep_prob_end / keep_prob_start)
        keep_prob = keep_prob_start * torch.exp(alpha * x)
    else:
        keep_prob = k * x + b
    keep_prob = torch.clamp(torch.tensor(keep_prob), min= keep_prob_start, max = keep_prob_end)
    keep_prob = keep_prob.reshape([])
    return keep_prob.float()


def get_st_global_scale(cfg, global_step, num_iter):
    num_steps = cfg.epochs*num_iter
    keep_prob_start = 0.0
    keep_prob_end = 1.0
    start_step = 0
    end_step = 0.1
    global_step = global_step.float()
    x = global_step / num_steps
    k = (keep_prob_end - keep_prob_start) / (end_step - start_step)
    b = keep_prob_start - k * start_step
    keep_prob = k * x + b
    keep_prob = torch.clamp(keep_prob, min = keep_prob_start, max = keep_prob_end)
    keep_prob = keep_prob.reshape([])
    return keep_prob.float32()


def align_predictions(outputs, alignment):
    outputs["points_1"] = q_rotate(outputs["points_1"], alignment)
    outputs["poses"] = q_mul(outputs["poses"], q_conj(alignment))
    outputs["pose_student"] = q_mul(outputs["pose_student"], q_conj(alignment))
    return outputs


def setup_sigma(cfg, global_step, num_iter):
    sigma_rel = get_smooth_sigma(cfg, global_step, num_iter)
    _sigma_rel = sigma_rel
    _gauss_sigma = sigma_rel / cfg.vox_size
    _gauss_kernel = smoothing_kernel(cfg, sigma_rel)
    return _sigma_rel, _gauss_sigma, _gauss_kernel


def replicate_for_multiview(tensor, cfg):
    new_tensor = tf_repeat_0(tensor, cfg.step_size)
    return new_tensor
   
def proj_loss_pose_candidates( gt, pred, inputs, cfg):
        """
        :param gt: [BATCH*VIEWS, IM_SIZE, IM_SIZE, 1]
        :param pred: [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        :return: [], [BATCH*VIEWS]
        """
        num_candidates = cfg.pose_predict_num_candidates
        gt = tf_repeat_0(gt, num_candidates) # [BATCH*VIEWS*CANDIDATES, IM_SIZE, IM_SIZE, 1]
        sq_diff = (gt - pred).mul(gt-pred)
        all_loss = torch.sum(sq_diff, [1, 2, 3]) # [BATCH*VIEWS*CANDIDATES]
        all_loss = torch.reshape(all_loss, [-1, num_candidates]) # [BATCH*VIEWS, CANDIDATES]
        min_loss = torch.min(all_loss, dim=1)[1] # [BATCH*VIEWS]
   
        min_loss_mask = torch.zeros(min_loss.shape[0], num_candidates).scatter_(1, min_loss.long().cpu().reshape(-1,1), 1) # [BATCH*VIEWS, CANDIDATES]
        num_samples = min_loss_mask.shape[0]

        min_loss_mask_flat = torch.reshape(min_loss_mask, [-1]) # [BATCH*VIEWS*CANDIDATES]
        min_loss_mask_final = torch.reshape(min_loss_mask_flat, [-1, 1, 1, 1]) # [BATCH*VIEWS*CANDIDATES, 1, 1, 1]
        loss_tensor = (gt - pred) * min_loss_mask_final.cuda()
        if cfg.variable_num_views:
            weights = inputs["valid_samples"]
            weights = tf_repeat_0(weights, num_candidates)
            weights = torch.reshape(weights, [weights.shape[0], 1, 1, 1])
            loss_tensor *= weights
        gt_tensor = gt * min_loss_mask_final.cuda()
        pred_tensor = pred * min_loss_mask_final.cuda()

        return gt_tensor, pred_tensor, min_loss

def add_student_loss(inputs, outputs, min_loss, cfg):
    num_candidates = cfg.pose_predict_num_candidates

    student = outputs['pose_student']
    teachers = outputs['poses']
    teachers = torch.reshape(teachers, [-1, num_candidates, 4])

    indices = min_loss
    indices = torch.unsqueeze(indices, dim = -1)
    batch_size = teachers.shape[0]
    batch_indices = torch.arange(0, batch_size, 1, dtype = torch.int).cuda()
    batch_indices = torch.unsqueeze(batch_indices, -1)
    indices = torch.cat([batch_indices.long(), indices], dim = 1)
    idx1, idx2 = indices.clone().detach().chunk(2,dim = -1)
    teachers = teachers[idx1,idx2].squeeze()
    teachers = Variable(teachers.data)

    if cfg.variable_num_views:
        weight = input['valid_samples']
    else:
        weight = 1.0

    if cfg.pose_student_align_loss:
        print('I did not realise it^^^^^^^smile')
    else:
        q_diff = q_norm(q_mul(teachers, q_conj(student)))
        angle_diff = q_diff[:, 0]
        student_loss = torch.sum((1.0 - angle_diff.mul(angle_diff) * weight))

    return student_loss

#!/usr/bin/env python

# import startup

import os

import numpy as np

import tensorflow as tf

from util.point_cloud_distance_tf import point_cloud_distance
from util.simple_dataset import Dataset3D
from util.app_config import config as app_config
from util.tools import partition_range, to_np_object
from util.quaternion_tf import quaternion_rotate
import scipy.io as scio
from scipy.io import loadmat


def compute_distance(cfg, sess, min_dist, idx, source, target, source_np, target_np):
    """
    compute projection from source to target
    """
    num_parts = cfg.pc_eval_chamfer_num_parts
    partition = partition_range(source_np.shape[0], num_parts)
    min_dist_np = np.zeros((0,))
    idx_np = np.zeros((0,))
    for k in range(num_parts):
        r = partition[k, :]
        src = source_np[r[0]:r[1]]
        (min_dist_0_np, idx_0_np) = sess.run([min_dist, idx],
                                             feed_dict={source: src,
                                                       target: target_np})
        min_dist_np = np.concatenate((min_dist_np, min_dist_0_np), axis=0)
        idx_np = np.concatenate((idx_np, idx_0_np), axis=0)
    return min_dist_np, idx_np


def run_eval():
    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )
    eval_unsup = False
    synth_set = '02958343'
    
    cfg = app_config

    gt_pred_points = {}
    gt_pred = {}

    eval_unsup = cfg.eval_unsupervised_shape

    exp_dir = 'predicted_point'
    num_views = cfg.num_views-1
    num_views = 1
    # pred = loadmat(f'/home/ncj/Desktop/GMP2020/Image2pc_semantic/predicted_point/{synth_set}pred_test_code_unsuper_test.mat')
    pred = loadmat(f'/home/ncj/Desktop/GMP2020/Image2pc_semantic/predicted_point/{synth_set}DPC_pred_test_code_test.mat')
    gt_val = loadmat(f'/home/ncj/Desktop/GMP2020/Image2pc_semantic/data/torch_data/{synth_set}gt_test.mat')
    out_dir = '/home/ncj/Desktop/GMP2020/Image2pc_semantic/predicted_point/' 
    # gt_dir = os.path.join(cfg.gt_pc_dir, cfg.synth_set)

    g = tf.Graph()
    
    with g.as_default():
        source_pc = tf.placeholder(dtype=tf.float64, shape=[None, 3])
        target_pc = tf.placeholder(dtype=tf.float64, shape=[None, 3])
        quat_tf = tf.placeholder(dtype=tf.float64, shape=[1, 4])

        _, min_dist, min_idx = point_cloud_distance(source_pc, target_pc)

        source_pc_2 = tf.placeholder(dtype=tf.float64, shape=[1, None, 3])
        rotated_pc = quaternion_rotate(source_pc_2, quat_tf)

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    save_pred_name = "{}_{}".format(cfg.save_predictions_dir, cfg.eval_split)
    save_dir = os.path.join(exp_dir, cfg.save_predictions_dir)

    if eval_unsup:
        reference_rotation = scipy.io.loadmat("{}/{}final_reference_rotation.mat".format(exp_dir, synth_set))["rotation"]

   
    num_models = len(pred['name'])

    model_names = []
    chamfer_dists = np.zeros((0, num_views, 2), dtype=np.float64)
    for k in range(num_models):

        # sample = pred['pose_gt'][k]

        print(f"{k}/{num_models}")
        print(pred['name'][k])

        all_pcs = pred["points"][k]

        has_number = False
        
        Vgt = gt_val['points'][0][k] 

        chamfer_dists_current = np.zeros((num_views, 2), dtype=np.float64)
        
        for i in range(num_views):
            pred_pcs = all_pcs[i, :, :]
            
            if eval_unsup:
                pred_pcs = np.expand_dims(pred_pcs, 0)
                pred_pcs = sess.run(rotated_pc, feed_dict={source_pc_2: pred_pcs,
                                                       quat_tf: reference_rotation})
                pred_pcs = np.squeeze(pred)

            gt_pred_points.setdefault('pred_rotate_points', []).append(pred_pcs)
            gt_pred_points.setdefault('gt_points', []).append(gt_val['points'][0][k])
            print('k')

            # pred_to_gt, idx_np = compute_distance(cfg, sess, min_dist, min_idx, source_pc, target_pc, pred_pcs, Vgt)
            # gt_to_pred, _ = compute_distance(cfg, sess, min_dist, min_idx, source_pc, target_pc, Vgt, pred_pcs)
            # chamfer_dists_current[i, 0] = np.mean(pred_to_gt)
            # chamfer_dists_current[i, 1] = np.mean(gt_to_pred)

            # is_nan = np.isnan(pred_to_gt)
            # assert(not np.any(is_nan))

    #     current_mean = np.mean(chamfer_dists_current, 0)
    #     print("total:", current_mean)
    #     chamfer_dists = np.concatenate((chamfer_dists, np.expand_dims(chamfer_dists_current, 0)))

    # final = np.mean(chamfer_dists, axis=(0, 1)) * 100
    # print(final)

    
    scio.savemat(out_dir + synth_set + 'pred_rotate_points_gt', {'gt_points': gt_pred_points['gt_points'], 'pred_rotate_points': gt_pred_points['pred_rotate_points']})  

    scio.savemat(os.path.join(exp_dir, "chamfer_{}.mat".format(save_pred_name)),
                     {"chamfer": chamfer_dists,
                      "model_names": to_np_object(model_names)})

    file = open(os.path.join(exp_dir, "chamfer_{}.txt".format(save_pred_name)), "w")
    file.write("{} {}\n".format(final[0], final[1]))
    file.close()


def main(_):
    run_eval()


if __name__ == '__main__':
    tf.app.run()

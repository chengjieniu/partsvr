# MIT License
#
# Copyright (c) 2018 Chen-Hsuan Lin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#import tensorflow as tf
import torch

def point_cloud_distance(Vs, Vt):
    """
    For each point in Vs computes distance to the closest point in Vt
    """
    VsN = Vs.shape[0]
    VtN = Vt.shape[0]
    Vt_rep = Vt[None, :, :].repeat([VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = Vs[:, None, :].repeat([1, VtN, 1])  # [VsN,VtN,3]
    diff = Vt_rep-Vs_rep
    dist = torch.sqrt(torch.sum(diff**2, axis=[2]))  # [VsN,VtN]
    idx = torch.argmin(dist, axis=1)
    #there are two lines code needs to be changed
    proj = tf.gather_nd(Vt_rep, torch.stack([tf.range(VsN), idx], axis=1))
    minDist = tf.gather_nd(dist, torch.stack([tf.range(VsN), idx], axis=1))
    return proj, minDist, idx

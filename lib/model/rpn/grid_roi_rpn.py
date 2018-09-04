## Zhenheng Yang
## 06/01/2018
## -------------------------------------------
## outputs a grid around the key_point as rois
## -------------------------------------------

from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .generate_anchors import generate_grid_anchors

import numpy as np
import math
import pdb
import time

class _GRID_ROI_RPN(nn.Module):
    """ region proposal generated from fix grid around the keypoint """
    def __init__(self, din):
        super(_GRID_ROI_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.grid_stride = cfg.GRID_STRIDE[0]
        self.grid_size = cfg.GRID_SIZE # [w, h]

        # define genuine generated proposals
        self.anchor_proposals = torch.from_numpy(generate_grid_anchors(scales=np.array(self.anchor_scales), 
                                    ratios=np.array(self.anchor_ratios))).float()
        self._num_anchors = self.anchor_proposals.size(0)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, kp_center):

        batch_size = base_feat.size(0)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        shift_x = np.arange(0, self.grid_size[0]) * self.grid_stride 
        shift_y = np.arange(0, self.grid_size[1]) * self.grid_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().float()
        kp_dist = torch.FloatTensor([self.grid_stride+0.5*32*self.anchor_scales[0],self.grid_stride+0.5*32*self.anchor_scales[0]]).expand(batch_size, 1, 2)
        kp_dist = kp_dist.cuda()
        kp_shift = (kp_center - kp_dist).repeat(1,1,2).contiguous()
        A = self._num_anchors
        K = kp_shift.size(1)
        self.anchor_proposals = self.anchor_proposals.float().cuda()
        anchors = self.anchor_proposals.view(1, A, 4).expand(batch_size, A, 4).contiguous()
        # shifts = shifts.expand(batch_size,K,4)
        anchors = anchors.view(batch_size, 1, A, 4) + kp_shift.view(batch_size, K, 1, 4)
        # anchors = anchors.view(batch_size, 1, A, 4) + shifts.view(batch_size, K, 1, 4)
        anchors = anchors.view(batch_size, K * A, 4)

        output = anchors.new(batch_size, K*A, 5).zero_()
        for i in range(batch_size):
            output[i,:,0] = i
            output[i,:,1:] = anchors[i,:,:]

        return output

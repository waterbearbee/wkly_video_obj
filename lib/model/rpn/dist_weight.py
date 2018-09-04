## Zhenheng Yang
## 07/01/2018
## ------------------------------------------------------------
## compute weights for proposals correponding to each keypoint
##
## input: proposal_centers of size [B, CONTEXT_NUM_ROIS, 2]
## 		  kp_center of size [B, MAX_NUM_GT_BOXES, 2]
##		  gt_rois_labels of size [B, MAX_NUM_GT_BOXES]
##		  kp_dist_mean of size [2, num_classes]
##		  kp_dist_var of size [2, num_classes]
## output: weight of each proposal for each kp_center
##			of size [B, CONTEXT_NUM_ROIS, MAX_NUM_GT_BOXES]
## ------------------------------------------------------------

from __future__ import absolute_import
import math
import torch
import pdb
from model.utils.config import cfg
from torch.autograd import Variable

def dist_weight(proposal_centers, kp_centers, gt_rois_labels, kp_dist_mean, kp_dist_var, gt_height, gt_width):

	assert(proposal_centers.size(1) == cfg.TRAIN.CONTEXT_NUM_ROIS)
	assert(kp_centers.size(1) == cfg.MAX_NUM_GT_BOXES)
	kp_dist_var_batch = kp_dist_var[:,gt_rois_labels.view(-1)].view(2,gt_rois_labels.size(0), gt_rois_labels.size(1))
	kp_dist_var_batch = kp_dist_var_batch.permute(1,2,0)
	kp_dist_mean_batch = kp_dist_mean[:,gt_rois_labels.view(-1)].view(2,gt_rois_labels.size(0), gt_rois_labels.size(1))
	kp_dist_mean_batch = kp_dist_mean_batch.permute(1,2,0)
	scaled_mean_x=kp_dist_mean_batch[:,:,0]*Variable(gt_height.expand(-1,5))
	scaled_mean_y=kp_dist_mean_batch[:,:,1]*Variable(gt_width.expand(-1,5))
	scaled_kp_dist_mean = torch.cat([scaled_mean_x.unsqueeze(2), scaled_mean_y.unsqueeze(2)], dim=2)
	dist_center_batch = Variable(kp_centers) + scaled_kp_dist_mean

	sigma_x, sigma_y = 10, 10
	proposal_centers = proposal_centers.unsqueeze(2).repeat(1,1,cfg.MAX_NUM_GT_BOXES,1)
	dist_center_batch = dist_center_batch.unsqueeze(1).repeat(1,cfg.TRAIN.CONTEXT_NUM_ROIS,1,1)
	center_diff = Variable(proposal_centers) - dist_center_batch #[B, num_rois, max_num_gt, 2]
	kp_dist_var_batch = kp_dist_var_batch.unsqueeze(1).repeat(1,cfg.TRAIN.CONTEXT_NUM_ROIS,1,1)

	proposal_normal_weights = 1./(math.pi*sigma_x*sigma_y)* \
					torch.exp(-0.5*(center_diff[:,:,:,0]**2/kp_dist_var_batch[:,:,:,0]**2+ \
						center_diff[:,:,:,1]**2/kp_dist_var_batch[:,:,:,1]**2))

	return proposal_normal_weights
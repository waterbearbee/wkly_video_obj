## Zhenheng Yang
## 06/10/2018
## -------------------------------------------------------------
## faster_rcnn adapted from Ross Girshick version
## takes person bounding box, proposal boxes, key point as input
## --------------------------------------------------------------

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.utils.find_grid_index import find_grid_index
from model.utils.action2object_label import action2object_label
from model.rpn.dist_weight import dist_weight
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_align.modules.roi_align import RoIAlignAvg
import time
import pdb

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, action_classes, object_classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.action_classes = action_classes
        self.n_action_classes = len(action_classes)
        self.obj_classes = object_classes
        self.n_obj_classes = len(object_classes)
        self.class_agnostic = class_agnostic
        self._action_class_to_ind = dict(zip(self.action_classes, xrange(self.n_action_classes)))
        self._action_ind_to_class = dict(zip(xrange(self.n_action_classes), self.action_classes))
        self._obj_class_to_ind = dict(zip(self.obj_classes, xrange(self.n_obj_classes)))
        self._obj_ind_to_class = dict(zip(xrange(self.n_obj_classes), self.obj_classes))
        # define pooling method
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

    def forward(self, im_data, im_info, gt_boxes, proposal_boxes, num_sec_boxes, num_kp, kp_centers, kp_dist_mean, kp_dist_var, kp_selection):

        batch_size = im_data.size(0)
        obj_cls_loss, action_cls_loss = 0, 0

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        proposal_boxes = proposal_boxes.data
        num_sec_boxes = num_sec_boxes.data
        num_kp = num_kp.data
        kp_centers = kp_centers.data #[B,5,2,17]
        kp_selection_soft = F.softmax(kp_selection/0.001,1).data # [num_actions, 17]
        gt_rois_label = gt_boxes[:,:,4].long().contiguous().view(-1)
        kp_selections = kp_selection_soft[gt_rois_label,:] #[2*5, 17]
        # pdb.set_trace()
        kp_centers = torch.sum(kp_centers*kp_selections.view(batch_size,-1,17).unsqueeze(2).repeat(1,1,2,1), dim=3)

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        rois = proposal_boxes.new(batch_size, proposal_boxes.size(1), 5).zero_()
        for i in range(batch_size):
            rois[i,:,0] = i
            rois[i,:,1:] = proposal_boxes[i,:,:4]
        rois_centers = torch.stack([0.5*(rois[:,:,1]+rois[:,:,3]), \
                        0.5*(rois[:,:,2]+rois[:,:,4])], dim=2)
        
        # rois are just proposals
        rois = Variable(rois)

        # do roi pooling for roi boxes
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # do roi pooling for ground truth boxes
        gt_rois_label = gt_boxes[:,:,4].long().contiguous().view(-1) # [B, 5] -> [B*5]
        gt_rois_label_onehot = torch.zeros(batch_size, self.n_action_classes).cuda().scatter_(1, gt_rois_label.view(batch_size,-1), 1.0)
        gt_rois = gt_boxes[:,[0],:] # shallow copy
        gt_height, gt_width = gt_rois[:,:,2]-gt_rois[:,:,0], gt_rois[:,:,3]-gt_rois[:,:,1] # [B,1]
        for i in range(batch_size):
            gt_rois[i,:,1:] = gt_rois[i,:,:4]
            gt_rois[i,:,0] = i
        gt_rois = Variable(gt_rois)
        gt_pooled_feat = self.RCNN_roi_align(base_feat, gt_rois.view(-1, 5))
        
        # feed pooled features to classification layers
        pooled_feat_fc7 = self._head_to_tail_sec(pooled_feat) # [B*700, 4096]
        gt_pooled_feat_fc7 = self._head_to_tail(gt_pooled_feat) #[B*1, 4096]

        # compute classification probability
        proposal_obj_cls_score = self.obj_cls_score(pooled_feat_fc7) # [B*700, num_object_class]
        # proposal_obj_det_score = self.obj_det_score(pooled_feat_fc7)
        proposal_action_cls_score = self.action_cls_score(pooled_feat_fc7) # [B*700, num_action_class]

        gt_cls_scores = self.action_cls_score(gt_pooled_feat_fc7) # [B, num_action_class]
        mil_cls_score = torch.max(proposal_action_cls_score.view(batch_size, -1, self.n_action_classes), 1)[0] #[B, num_action_classes]
        sum_cls_scores = mil_cls_score + gt_cls_scores
        sum_cls_prob = F.sigmoid(sum_cls_scores).clamp(0,1)

        rois_label = Variable(gt_rois_label_onehot.clamp(0,1))

        if self.training:
            # compute probability from bivariate normal distribution
            proposal_weights = dist_weight(rois_centers, kp_centers, gt_rois_label.view(batch_size,-1),
                         kp_dist_mean, kp_dist_var, gt_height, gt_width) #[B, 700, 5]
            soft_temp = 10.0
            proposal_weights = F.softmax(proposal_weights, dim=1)*10

            # action classification loss
            action_cls_loss = F.binary_cross_entropy(sum_cls_prob, rois_label)

            # object classification loss
            ## WSDDN header
            proposal_obj_cls_score = proposal_obj_cls_score.view(batch_size,-1,self.n_obj_classes) #[B,700,17]
            # delta_cls = F.softmax(proposal_obj_cls_score, dim=-1)
            # delta_det = F.softmax(proposal_obj_cls_score, dim=1)
            # scores = (delta_cls*delta_det)
            # scores_cls = torch.sum(scores, dim=1).clamp(0,1)
            scores = F.sigmoid(proposal_obj_cls_score)
            scores_cls = (torch.sum(scores, dim=1)/cfg.TRAIN.CONTEXT_NUM_ROIS).clamp(0,1)
            proposal_obj_labels = action2object_label(gt_rois_label, self._action_ind_to_class, self._obj_class_to_ind) #[B,700,5,17] #[B, 17]
            
            # print("------------")
            # print(scores_cls)
            # print('max value: {}'.format(scores_cls.max()))

            # print('min value: {}'.format(scores_cls.min()))

            obj_cls_loss = F.binary_cross_entropy(scores_cls, proposal_obj_labels)
            # print(obj_cls_loss)
            # change from BCE to soft label cross entropy
            # (1-l)log(1-s) + l log(s) --> (1-w)log(1-s) + wlog(s)
            # proposal_obj_labels = proposal_obj_labels*proposal_weights.unsqueeze(3).expand(-1,-1,-1,self.n_obj_classes)

            # # convert proposal_cls_score to probability
            # proposal_obj_cls_prob = F.sigmoid(proposal_obj_cls_score) # [B*700,17]

            # for i in xrange(proposal_obj_labels.size(-2)):
                
            #     print(obj_cls_loss) 


                # obj_cls_loss += -1*torch.mean(torch.log(1-proposal_obj_cls_prob+1e-7)*(1-proposal_obj_labels[:,:,i,:].contiguous().view(-1, self.n_obj_classes)) + 
                #             torch.log(proposal_obj_cls_prob+1e-7)*(proposal_obj_labels[:,:,i,:].contiguous().view(-1, self.n_obj_classes)), dim=1) * \
                #             proposal_weights[:,:,i].contiguous().view(-1)
                # obj_cls_loss += torch.mean(F.binary_cross_entropy(
                #             proposal_obj_cls_score, # [B*700, 17]
                #             proposal_obj_labels[:,:,i,:].contiguous().view(-1, self.n_obj_classes), size_average=False) #[B,700,5,17]
                #             * proposal_weights[:,:,i].contiguous().view(-1))

        return action_cls_loss, obj_cls_loss, proposal_obj_cls_score, proposal_obj_cls_score, proposal_weights, rois_label, kp_centers

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.obj_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.action_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

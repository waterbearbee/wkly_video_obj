# ---------------------------------------------------------
# Zhenheng Yang, 06/14/2018
# For loading data in batch and randomly rescale/crop image
# ---------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch

import numpy as np
import numpy.random as npr
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_action_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_action_classes = num_action_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.max_num_sec_box = cfg.TRAIN.CONTEXT_NUM_ROIS
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_action_classes)
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)
    if self.training:
        # blobs['gt_boxes'], blobs['key_points'] = self.unison_shuffle(blobs['gt_boxes'], blobs['key_points']) 
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        sec_roi_boxes = torch.from_numpy(blobs['sec_roi_boxes'])
        key_points = torch.from_numpy(blobs['key_points'])
        img_id = blobs['img_id']
        img_name = blobs['img_name']

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index]

        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height                
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)
                sec_roi_boxes[:,1] = sec_roi_boxes[:,1] - float(y_s)
                sec_roi_boxes[:,3] = sec_roi_boxes[:,3] - float(y_s)
                key_points[:,1,:] = key_points[:,1,:] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)
                sec_roi_boxes[:,1].clamp_(0, trim_size-1)
                sec_roi_boxes[:,3].clamp_(0, trim_size-1)
                key_points[:,1,:].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width                
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                sec_roi_boxes[:,0] = sec_roi_boxes[:,0] - float(x_s)
                sec_roi_boxes[:,2] = sec_roi_boxes[:,2] - float(x_s)
                key_points[:,0,:] = key_points[:,0,:] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)
                sec_roi_boxes[:,0].clamp_(0, trim_size-1)
                sec_roi_boxes[:,2].clamp_(0, trim_size-1)
                key_points[:,0,:].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()

            padding_data[:data_height, :, :] = data[0]
            # update im_info
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            # gt_boxes.clamp_(0, trim_size)
            gt_boxes[:, :4].clamp_(0, trim_size)
            sec_roi_boxes[:, :4].clamp_(0, trim_size)
            key_points.clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size


        # # check the bounding box:
        # not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        # keep = torch.nonzero(not_keep == 0).view(-1)

        # assert gt_boxes [action_clses,5], key_points [NUM_GT_BOX,2], SEC_BOX_ROI [CONTEXT_NUM_ROIS,5]
        padding_gt_boxes = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        padding_kp = torch.FloatTensor(self.max_num_box, key_points.size(1), 17).zero_()
        padding_sec_roi_boxes = torch.FloatTensor(self.max_num_sec_box, sec_roi_boxes.size(1)).zero_()
        num_sec_boxes = min(sec_roi_boxes.size(0), self.max_num_sec_box)
        num_kp = min(key_points.size(0), self.max_num_box)

        # random sampling or padding the sec_roi_boxes
        if sec_roi_boxes.size(0)> self.max_num_sec_box:
            cinds = npr.choice(np.arange(sec_roi_boxes.size(0)), size=self.max_num_sec_box,
                                                replace=False)
        elif sec_roi_boxes.size(0) > 0:
            cinds = npr.choice(np.arange(sec_roi_boxes.size(0)), size=self.max_num_sec_box,
                                                replace=True)
        assert(cinds.size == self.max_num_sec_box),"Secondary RoIs are not of correct size"

        # random sampling or padding the key_points
        if key_points.size(0)> self.max_num_box:
            kinds = npr.choice(np.arange(key_points.size(0)), size=self.max_num_box,
                                                replace=False)
        elif key_points.size(0) > 0:
            kinds = npr.choice(np.arange(key_points.size(0)), size=self.max_num_box,
                                                replace=True)
        assert(kinds.size == self.max_num_box),"Key_points are not of correct size"

        if gt_boxes.size(0)> self.max_num_box:
            ginds = npr.choice(np.arange(gt_boxes.size(0)), size=self.max_num_box,
                                                replace=False)
        elif gt_boxes.size(0) > 0:
            ginds = npr.choice(np.arange(gt_boxes.size(0)), size=self.max_num_box,
                                                replace=True)
        assert(ginds.size == self.max_num_box),"Gt_boxes are not of correct size"

        # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)
        padding_sec_roi_boxes = sec_roi_boxes[cinds]
        padding_kp = key_points[kinds]
        padding_gt_boxes = gt_boxes[ginds]

        return padding_data, im_info, padding_gt_boxes, padding_sec_roi_boxes, padding_kp, num_sec_boxes, num_kp, img_name
    else:
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        sec_roi_boxes = torch.from_numpy(blobs['sec_roi_boxes'])
        key_points = torch.from_numpy(blobs['key_points'])
        img_id = blobs['img_id']
        img_name = blobs['img_name']
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)
        num_sec_boxes = sec_roi_boxes.size(1)
        num_kp = 0

        return data, im_info, gt_boxes, sec_roi_boxes, key_points, num_sec_boxes, num_kp, img_name

  def unison_shuffle(self, a, b):
    assert(len(a) == len(b))
    p = npr.permutation(len(a))
    return a[p], b[p]


  def __len__(self):
    return len(self._roidb)

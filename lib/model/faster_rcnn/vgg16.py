# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------

## -------------------------------------------------------
## modified for one fc layer and two sibling cls layers
## -------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, action_classes, obj_classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, action_classes, obj_classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-4])
    vgg.classifier_sec = nn.Sequential(
                        nn.Linear(512 * 7 * 7, 4096),
                        nn.ReLU(True),
                        nn.Dropout())

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the conv layers:
    print("length of rcnn base model")
    print(len(self.RCNN_base))
    for layer in range(30):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    self.RCNN_top = vgg.classifier
    self.RCNN_top_sec = vgg.classifier_sec
    for ind in range(len(self.RCNN_top_sec)):
        for p in self.RCNN_top_sec[ind].parameters(): p.requires_grad=False

    # not using the last maxpool layer
    self.obj_cls_score = nn.Linear(4096, self.n_obj_classes)
    self.action_cls_score = nn.Linear(4096, self.n_action_classes)
    for p in self.obj_cls_score.parameters(): p.requires_grad=False     

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

  def _head_to_tail_sec(self, pool5):

    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top_sec(pool5_flat)

    return fc7


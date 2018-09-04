## Zhenheng Yang
## 05/24/2018
## --------------------------------------------------------------------------------------
## adapated from https://github.com/jwyang/faster-rcnn.pytorch/blob/master/trainval_net.py
## --------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import (
    weights_normal_init,
    save_net,
    load_net,
    adjust_learning_rate,
    save_checkpoint,
    clip_gradient,
)

from model.nms.nms_wrapper import nms

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="pascal_voc",
        type=str,
    )
    parser.add_argument(
        "--net", dest="net", help="vgg16, res101", default="vgg16", type=str
    )
    parser.add_argument(
        "--start_epoch", dest="start_epoch", help="starting epoch", default=1, type=int
    )
    parser.add_argument(
        "--epochs",
        dest="max_epochs",
        help="number of epochs to train",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--disp_interval",
        dest="disp_interval",
        help="number of iterations to display",
        default=40,
        type=int,
    )
    parser.add_argument(
        "--checkpoint_interval",
        dest="checkpoint_interval",
        help="number of iterations to display",
        default=10000,
        type=int,
    )

    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="directory to save models",
        default="/home/zhenheng/Documents/",
        type=str,
    )
    parser.add_argument(
        "--load_dir",
        dest="load_dir",
        help="directory to load pre-train models",
        default="/home/zhenheng/Documents/",
        type=str,
    )
    parser.add_argument(
        "--nw",
        dest="num_workers",
        help="number of worker to load data",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "--ls",
        dest="large_scale",
        help="whether use large imag scale",
        action="store_true",
    )
    parser.add_argument(
        "--mGPUs", dest="mGPUs", help="whether use multiple GPUs", action="store_true"
    )
    parser.add_argument(
        "--bs", dest="batch_size", help="batch_size", default=1, type=int
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )

    # config optimization
    parser.add_argument(
        "--o", dest="optimizer", help="training optimizer", default="adam", type=str
    )
    parser.add_argument(
        "--lr", dest="lr", help="starting learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--lr_decay_step",
        dest="lr_decay_step",
        help="step to do learning rate decay, unit is epoch",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--lr_decay_gamma",
        dest="lr_decay_gamma",
        help="learning rate decay ratio",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--s", dest="session", help="training session", default=1, type=int
    )
    parser.add_argument(
        "--r", dest="resume", help="resume checkpoint or not", default=0, type=int
    )
    parser.add_argument(
        "--checksession",
        dest="checksession",
        help="checksession to load model",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkepoch",
        dest="checkepoch",
        help="checkepoch to load model",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint to load model",
        default=0,
        type=int,
    )
    # log and diaplay
    parser.add_argument(
        "--use_tfboard",
        dest="use_tfboard",
        help="whether use tensorflow tensorboard",
        default=1,
        type=int,
    )
    # subpath to store the tfboard summary
    parser.add_argument(
        "--tfboard_path",
        dest="tfboard_path",
        help="subpath in logs folder to save the tfboard summary",
        default="",
        type=str,)

    # weighting terms for action and object classification loss
    parser.add_argument(
        "--obj_cls_wt",
        dest="obj_cls_wt",
        help="weights for object classification loss",
        default=1.0,
        type=float)
    parser.add_argument(
        "--act_cls_wt",
        dest="act_cls_wt",
        help="weights for action classification loss",
        default=1.0,
        type=float)
    # evaluation per epoch
    # parser.add_argument(
    #     "--eval_per_epoch",
    #     dest="eval_per_epoch",
    #     help="whether evaluation per epoch",
    #     default=False,
    #     type=bool,
    # )

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(
                self.num_per_batch * batch_size, train_size
            ).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = (
            rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        )

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == "__main__":

    args = parse_args()

    print("Called with args:")
    print(args)

    if args.use_tfboard:
        from model.utils.logger import Logger

        # Set the logger
        logger = Logger("./logs/"+args.tfboard_path)

    if args.dataset == "charades":
        args.imdb_name = "charades_train"
        args.imdbval_name = "charades_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[1]",
            "ANCHOR_RATIOS",
            "[1]",
            "MAX_NUM_GT_BOXES",
            "5",
        ]

    args.cfg_file = (
        "cfgs/{}_ls.yml".format(args.net)
        if args.large_scale
        else "cfgs/{}.yml".format(args.net)
    )

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print("{:d} roidb entries for training".format(len(roidb)))

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(
        roidb, ratio_list, ratio_index, args.batch_size, imdb.num_action_classes, training=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler_batch,
        num_workers=args.num_workers,
    )

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_sec_boxes = torch.LongTensor(1)
    num_kp = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    proposal_boxes = torch.FloatTensor(1)
    kp_center = torch.FloatTensor(1)
    kp_dist_mean = torch.zeros([2, imdb.num_action_classes]) # size of [2. num_class], 2 indicates [x,y]
    kp_dist_var = torch.ones([2, imdb.num_action_classes])*0.1
    kp_selection = torch.randn(imdb.num_action_classes, 17)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_sec_boxes = num_sec_boxes.cuda()
        num_kp = num_kp.cuda()
        gt_boxes = gt_boxes.cuda()
        proposal_boxes = proposal_boxes.cuda()
        kp_center = kp_center.cuda()
        kp_dist_mean = kp_dist_mean.cuda()
        kp_dist_var = kp_dist_var.cuda()
        kp_selection = kp_selection.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_sec_boxes = Variable(num_sec_boxes)
    num_kp = Variable(num_kp)
    gt_boxes = Variable(gt_boxes)
    proposal_boxes = Variable(proposal_boxes)
    kp_center = Variable(kp_center)
    kp_dist_mean = Variable(kp_dist_mean, requires_grad=True)
    kp_dist_var = Variable(kp_dist_var, requires_grad=True)
    kp_selection = Variable(kp_selection, requires_grad=True)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    pretrained_flag = True
    if args.resume:
        pretrained_flag = False
    if args.net == "vgg16":
        fasterRCNN = vgg16(
            imdb._action_classes, imdb._obj_classes, pretrained=pretrained_flag, class_agnostic=args.class_agnostic
        )
    elif args.net == "res101":
        fasterRCNN = resnet(
            imdb._action_classes, imdb._obj_classes, 101, pretrained=pretrained_flag, class_agnostic=args.class_agnostic
        )
    elif args.net == "res50":
        fasterRCNN = resnet(
            imdb.classes, 50, pretrained=pretrained_flag, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            print(key)
            if "bias" in key:
                params += [
                    {
                        "params": [value],
                        "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                        "weight_decay": cfg.TRAIN.BIAS_DECAY
                        and cfg.TRAIN.WEIGHT_DECAY
                        or 0,
                    }
                ]
            else:
                params += [
                    {
                        "params": [value],
                        "lr": lr,
                        "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
                    }
                ]
    params += [
            {
            "params": [kp_dist_mean],
            "lr": 10*lr,
            "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
            }
            ]
    params += [
            {
            "params": [kp_dist_var],
            "lr": 10*lr,
            "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
            }
            ]
    params += [
            {
            "params": [kp_selection],
            "lr": 10*lr,
            "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
            }
            ]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(
            args.load_dir,
            "faster_rcnn_{}_{}_{}.pth".format(
                args.checksession, args.checkepoch, args.checkpoint
            ),
        )
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint["session"]
        args.start_epoch = checkpoint["epoch"]
        fasterRCNN.load_state_dict(checkpoint["model"])
        # pdb.set_trace()
        # optimizer.load_state_dict(checkpoint["optimizer"])
        lr = optimizer.param_groups[0]["lr"]
        if "pooling_mode" in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    if args.cuda:
        fasterRCNN.cuda()

    iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        loss_act_temp, loss_obj_temp = 0, 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        # data_iter_test = iter(dataloader_test)

        for step in range(iters_per_epoch):
            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            proposal_boxes.data.resize_(data[3].size()).copy_(data[3])
            kp_center.data.resize_(data[4].size()).copy_(data[4])
            num_sec_boxes.data.resize_(data[5].size()).copy_(data[5])
            num_kp.data.resize_(data[6].size()).copy_(data[6])

            fasterRCNN.zero_grad()
            # pdb.set_trace()
            act_cls_loss, obj_cls_loss, _, _, proposal_weights, labels_cls, kp_centers = fasterRCNN(im_data, im_info, gt_boxes, proposal_boxes,
                                                num_sec_boxes, num_kp, kp_center, kp_dist_mean, kp_dist_var, kp_selection)
            weighted_obj_cls_loss = args.obj_cls_wt * obj_cls_loss
            weighted_act_cls_loss = args.act_cls_wt * act_cls_loss
            loss = (
                weighted_act_cls_loss.mean() +
                weighted_obj_cls_loss.mean()
            )
            loss_temp += loss.data[0]
            loss_act_temp += weighted_act_cls_loss.data[0]
            loss_obj_temp += weighted_obj_cls_loss.data[0]

            # backward
            optimizer.zero_grad()
            # pdb.set_trace()
            loss.backward()
            # pdb.set_trace()
            # if args.net == "vgg16":
            #     clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            weighted_action_cls_loss = weighted_act_cls_loss.data[0]
            weighted_obj_cls_loss = weighted_obj_cls_loss.data[0] 

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval
                    loss_act_temp /= args.disp_interval
                    loss_obj_temp /= args.disp_interval
                    print(
                            "[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                            % (args.session, epoch, step, iters_per_epoch, loss_temp, lr)
                        )

                    print(
                        "\t\t\taction_cls: %.4f; object_cls: %.4f" % (weighted_action_cls_loss, weighted_obj_cls_loss)
                    )
                    # print(scores_cls)
                    # print(labels_cls)
                if args.use_tfboard:
                    info = {
                        "loss": loss_temp,
                        "loss_action_cls": loss_act_temp,
                        "loss_obj_cls": loss_obj_temp
                    }
                    img_bbox = []
                    for i in range(args.batch_size):
                        img = np.array(data[0][i].permute(1,2,0)) + cfg.PIXEL_MEANS
                        img = img[:,:,::-1].copy()
                        # pdb.set_trace()
                        kp_centers_b = kp_centers[i]
                        proposal_weight = proposal_weights.data.cpu()[i,:,:]
                        proposal_box = proposal_boxes.data.cpu()[i,:,:]

                        top_k_props = np.argsort(-1*proposal_weight,0)
                        height, width = gt_boxes[i][0][3]-gt_boxes[i][0][1], gt_boxes[i][0][2]-gt_boxes[i][0][0]
                        for j in range(len(kp_centers_b)):
                            closest_box = proposal_box[top_k_props[0][j],:]
                            cv2.rectangle(img, (int(closest_box[0]), int(closest_box[1])), (int(closest_box[2]), int(closest_box[3])), (255,0,0))
                            cv2.circle(img, (int(kp_centers_b[j,0]), int(kp_centers_b[j,1])), 10, (0,255,0), thickness=-1)
                        cv2.rectangle(img, (int(data[2][i][0][0]), int(data[2][i][0][1])), (int(data[2][i][0][2]), int(data[2][i][0][3])), (0,0,255), thickness=2)
                        cv2.putText(img, data[7][i], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        action_classes = list(data[2][i][:,4])
                        kp_dist_centers = np.array(kp_dist_mean.cpu().data)[:,np.array(action_classes,dtype=np.int16)]
                        for kp_ind in range(kp_dist_centers.shape[1]):
                            cv2.circle(img, (int(kp_dist_centers[0,kp_ind]+kp_centers_b[kp_ind,0]),int(kp_dist_centers[1,kp_ind]+kp_centers_b[kp_ind,1])), 10, (0,0,255), thickness=-1)
                        action_classes = list(set(action_classes))
                        for j in range(len(action_classes)):
                            cv2.putText(img, imdb._action_names[int(action_classes[j])], (0, img.shape[0]-30*(j)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        # for roi in rois:
                        #     cv2.rectangle(img, (int(roi[1]), int(roi[2])), (int(roi[3]), int(roi[4])), (255,0,0), thickness=2)
                        img_bbox.append(img)
                    info['image_kp_bbox'] = np.array(img_bbox)


                    for tag, value in info.items():
                        if "loss" in tag:
                            logger.scalar_summary(tag, value, step+epoch*iters_per_epoch)
                        elif "image" in tag:
                            logger.image_summary(tag, value, step+epoch*iters_per_epoch)

                loss_temp = 0
                loss_act_temp = 0
                loss_obj_temp = 0
                start = time.time()


        if args.mGPUs:
            save_name = os.path.join(
                output_dir, "faster_rcnn_{}_{}_{}.pth".format(args.session, epoch, step)
            )
            save_checkpoint(
                {
                    "session": args.session,
                    "epoch": epoch + 1,
                    "model": fasterRCNN.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "kp_dist_mean": kp_dist_mean,
                    "kp_dist_var": kp_dist_var,
                    "kp_selection": kp_selection,
                    "pooling_mode": cfg.POOLING_MODE,
                    "class_agnostic": args.class_agnostic,
                },
                save_name,
            )
        else:
            save_name = os.path.join(
                output_dir, "faster_rcnn_{}_{}_{}.pth".format(args.session, epoch, step)
            )
            save_checkpoint(
                {
                    "session": args.session,
                    "epoch": epoch + 1,
                    "model": fasterRCNN.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "kp_dist_mean": kp_dist_mean,
                    "kp_dist_var": kp_dist_var,
                    "kp_selection": kp_selection,
                    "pooling_mode": cfg.POOLING_MODE,
                    "class_agnostic": args.class_agnostic,
                },
                save_name,
            )
        print("save model: {}".format(save_name))

        end = time.time()
        print(end - start)

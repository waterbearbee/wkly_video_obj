## Zhenheng Yang
## 07/05/2018
## --------------------------------------------------------------------------------------
## Test saved network for object detection performance evaluation
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
import torch.nn.functional as F
import pickle
from datasets.charades_eval import charades_eval
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.nms.nms_cpu import nms_cpu
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


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
        "--cfg",
        dest="cfg_file",
        help="optional config file",
        default="cfgs/vgg16.yml",
        type=str,
    )
    parser.add_argument(
        "--net",
        dest="net",
        help="vgg16, res50, res101, res152",
        default="res101",
        type=str,
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--load_dir",
        dest="load_dir",
        help="directory to load models",
        default="trained_models",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        help="directory to the evaluating model inside load_dir",
        default="",
        type=str,
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
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )
    parser.add_argument(
        "--parallel_type",
        dest="parallel_type",
        help="which part of model to parallel, 0: all, 1: model before roi pooling",
        default=0,
        type=int,
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
        help="checkepoch to load network",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint to load network",
        default=10021,
        type=int,
    )
    parser.add_argument(
        "--bs", dest="batch_size", help="batch_size", default=1, type=int
    )
    parser.add_argument(
        "--vis", dest="vis", help="visualization mode", default=0, type=int
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    print("Called with args:")
    print(args)

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "charades":
        args.imdb_name = "charades_train"
        args.imdbval_name = "charades_test"
        args.set_cfgs = ["ANCHOR_SCALES", "[1]", "ANCHOR_RATIOS", "[1]"]

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

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)

    print("{:d} roidb entries".format(len(roidb)))

    input_dir = args.load_dir + "/" + args.model_path
    load_name = os.path.join(
        input_dir,
        "faster_rcnn_{}_{}_{}.pth".format(
            args.checksession, args.checkepoch, args.checkpoint
        ),
    )

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(
            imdb._action_classes, imdb._obj_classes, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.net == "res101":
        fasterRCNN = resnet(
            imdb._action_classes, imdb._obj_classes, 101, pretrained=False, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint["model"])
    kp_dist_mean = checkpoint["kp_dist_mean"]
    kp_dist_var = checkpoint["kp_dist_var"]
    kp_selection = checkpoint["kp_selection"]

    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]

    print("load model successfully!")
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_sec_boxes = torch.LongTensor(1)
    num_kp = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    proposal_boxes = torch.FloatTensor(1)
    kp_center = torch.FloatTensor(1)
    # kp_dist_mean = torch.zeros([2, imdb.num_action_classes]) # size of [2. num_class], 2 indicates [x,y]

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_sec_boxes = num_sec_boxes.cuda()
        num_kp = num_kp.cuda()
        gt_boxes = gt_boxes.cuda()
        proposal_boxes = proposal_boxes.cuda()
        kp_center = kp_center.cuda()
        # kp_dist_mean = kp_dist_mean.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_sec_boxes = Variable(num_sec_boxes)
    num_kp = Variable(num_kp)
    gt_boxes = Variable(gt_boxes)
    proposal_boxes = Variable(proposal_boxes)
    kp_center = Variable(kp_center)
    # kp_dist_mean = Variable(kp_dist_mean, requires_grad=True)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis
    print("visualization: "+str(vis))

    thresh = 0.0

    save_name = "faster_rcnn_12"
    num_images = len(roidb)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_obj_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(
        roidb,
        ratio_list,
        ratio_index,
        args.batch_size,
        imdb.num_action_classes,
        training=False,
        normalize=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    data_iter = iter(dataloader)

    _t = {"im_detect": time.time(), "misc": time.time()}
    det_file = os.path.join(output_dir, "detections.pkl")

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    
    # print("Evaluating detections")
    # aps = []
    # corlocs = []
    # detpath = imdb._data_path+'/results/ws_det_' +imdb._image_set +'_{:s}.txt'
    # annopath = imdb._data_path+'/ImageSets/annotations.pkl'
    # imagelistfile= imdb._data_path+'/ImageSets/charades_action_cls_test.txt'
    # with open(annopath, 'rb') as f:
    #     try:
    #         recs = pickle.load(f)
    #     except:
    #         recs = pickle.load(f, encoding='bytes')
    # for cls in imdb._obj_classes:
    #     if cls == "__BACKGROUND__": continue
    #     rec, prec, ap, corloc = charades_eval(detpath, recs, imagelistfile, cls, ovthresh=0.5)
    #     aps += [ap]
    #     corlocs += [corloc]
    #     print('AP for {} = {:.4f}'.format(cls, ap))
    #     print('CorLoc for {} = {:.4f}'.format(cls, corloc))
    # print('Mean AP = {:.4f}'.format(np.mean(aps)))
    # print('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))

    for i in range(num_images):

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        proposal_boxes.data.resize_(data[3].size()).copy_(data[3])
        kp_center.data.resize_(data[4].size()).copy_(data[4])
        num_sec_boxes.data.resize_(data[5].size()).copy_(data[5])
        num_kp.data.resize_(data[6].size()).copy_(data[6])

        det_tic = time.time()
        _, _, prop_obj_cls_scores, prop_obj_det_scores, _, _, kp_centers = fasterRCNN(im_data, im_info, gt_boxes, 
                                        proposal_boxes, num_sec_boxes, num_kp, kp_center, kp_dist_mean, kp_dist_var, kp_selection) 
        # prop_obj_cls_scores: [B*num_sec_rois, num_obj_classes]
        prop_obj_cls_scores = prop_obj_cls_scores.view(args.batch_size, -1, imdb.num_obj_classes)
        pred_boxes = proposal_boxes.cpu().data.unsqueeze(2).expand(-1,-1,imdb.num_obj_classes,-1)[:,:,:,:4] # [B, 700, 17, 4]
        # delta_cls = F.softmax(prop_obj_cls_scores, dim=-1).cpu().data
        # delta_det = F.softmax(prop_obj_det_scores, dim=1).cpu().data
        scores = F.sigmoid(prop_obj_cls_scores) # [B, 700, num_obj_classes]
        pred_boxes = pred_boxes/data[1][0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_obj_classes):
            inds = torch.nonzero(scores[:,j] >= thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds.data]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = torch.FloatTensor(np.array(pred_boxes)[inds.data,j,:])
                cls_dets = torch.cat((cls_boxes, cls_scores.data.cpu().unsqueeze(1)), 1)
                cls_dets = cls_dets[order.data.cpu()]
                keep = nms_cpu(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(
                        im2show, imdb._obj_classes[j], cls_dets.cpu().numpy(), 0.3
                    )
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack(
                [all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_obj_classes)]
            )
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_obj_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write(
            "im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r".format(
                i + 1, num_images, detect_time, nms_time
            )
        )
        sys.stdout.flush()
        if vis:
            if not os.path.exists('visualization/{:s}'.format(args.net)):
                os.mkdir('visualization/{:s}'.format(args.net))
            cv2.imwrite("visualization/{:s}/{:s}.png".format(args.net, os.path.basename(imdb.image_path_at(i))), im2show)

    # pickle.dump(all_boxes, open('res/temp/detections.pkl', 'wb'))

    imdb._write_results_file_charades(all_boxes)

    print("Evaluating detections")
    aps = []
    corlocs = []
    detpath = imdb._data_path+'/results/ws_det_' +imdb._image_set +'_{:s}.txt'
    annopath = imdb._data_path+'/ImageSets/annotations.pkl'
    imagelistfile= imdb._data_path+'/ImageSets/charades_action_cls_test.txt'
    with open(annopath, 'rb') as f:
        try:
            recs = pickle.load(f)
        except:
            recs = pickle.load(f, encoding='bytes')
    for cls in imdb._obj_classes:
        if cls == "__BACKGROUND__": continue
        rec, prec, ap, corloc = charades_eval(detpath, recs, imagelistfile, cls, ovthresh=0.5)
        aps += [ap]
        corlocs += [corloc]
        print('AP for {} = {:.4f}'.format(cls, ap))
        print('CorLoc for {} = {:.4f}'.format(cls, corloc))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))

    end = time.time()
    print("test time: %0.4fs" % (end - start))

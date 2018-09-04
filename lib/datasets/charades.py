## Zhenheng Yang
## 06/03/2018
## ---------------------------------------
# For loading charades dataset into imdb
# 06/10
# Add loading keypoints from pkl files
## ---------------------------------------

# from __future__ import print_function
from __future__ import absolute_import
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
import cPickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
import model.utils.cython_bbox
import pdb


class charades(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'charades_' + image_set)
        self._year = '2017'
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._base_path = self._devkit_path
        self._data_path = self._devkit_path
        self._action_classes = []
        self._action_names = []
        with open(os.path.join(devkit_path, 'annotation', 'Charades_v1_classes_17obj.txt')) as f:
            lines = f.readlines()
            for line in lines:
                self._action_names.append(line.rstrip())
                self._action_classes.append(line.split(" ")[0])
        self._obj_classes = []
        with open(os.path.join(self._data_path, 'ImageSets', 'charade_object_classes.txt')) as f:
            lines = f.readlines()
            for line in lines:
                self._obj_classes.append(line.rstrip())
        self._action_classes = tuple(self._action_classes)
        self._obj_classes = tuple(self._obj_classes)
        self._action_class_to_ind = dict(zip(self._action_classes, xrange(self.num_action_classes)))
        self._action_ind_to_class = dict(zip(xrange(self.num_action_classes), self._action_classes))
        self._obj_class_to_ind = dict(zip(self._obj_classes, xrange(self.num_obj_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True}

        assert os.path.exists(self._devkit_path), \
                'Charades data path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._base_path), \
                'Base path does not exist: {}'.format(self._base_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._base_path, 'JPEGImages',
                                  index.split(".")[0] + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        if self._image_set == 'train':
            image_set_file = os.path.join(self._devkit_path, 'ImageSets',
                                        'charadesdet_annotation_list.txt')
        else:
            image_set_file = os.path.join(self._devkit_path, 'ImageSets',
                                        'charades_action_cls_test.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.rstrip().split(' ')[0] for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where the charades data should be.
        """
        return os.path.join(datasets.ROOT_DIR)

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb_kpsel.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print('{:s} gt roidb loaded from {:s}'.format(self.name, cache_file))
        #     return roidb

        # Load all annotation file data (should take < 30 s).
        gt_roidb = self._load_charades_annotation()

        # print number of ground truth classes
        cc = np.zeros(len(self._action_classes), dtype = np.int16)
        for i in xrange(len(gt_roidb)):
            for n in xrange(len(gt_roidb[i]['gt_classes'])):
                cc[gt_roidb[i]['gt_classes'][n]] +=1

        for ic,nc in enumerate(cc):
            print "Count {:s} : {:d}".format(self._action_classes[ic], nc)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {:s}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                 self.name + '_' + self._image_set + '_ss_roidb_kpsel.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            new_img_indexes = []
            for i in range(len(roidb)):
                new_img_indexes.append(roidb[i]['img_index'])
            self._image_index = new_img_indexes
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = self._merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _merge_roidbs(self, a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            # a[i]['gt_overlaps'] = scipy.sparse.csr_matrix(np.vstack([a[i]['gt_overlaps'],
            #                                                         b[i]['gt_overlaps']]))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            if 'key_points' in b[i]:
                a[i]['key_points'] = b[i]['key_points']
        return a

    def _load_selective_search_roidb(self, gt_roidb):

        if self._image_set == 'train':
            filename = os.path.join(self.cache_path, 'selective_search_test_data_charades.mat')
        else:
            filename = os.path.join(self.cache_path, 'selective_search_action_cls_test_charades.mat')

        assert os.path.exists(filename), \
                'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)
        # raw_data = pickle.load(open(filename, 'rb'))

        # num_images = len(raw_data)
        ss_roidb = []

        for i in xrange(len(gt_roidb)):

            boxes = raw_data['loadres'][0,0][1].ravel()[self.new_index[i]][:, (1, 0, 3, 2)] - 1
            num_boxes = boxes.shape[0]

            gt_boxes = gt_roidb[i]['boxes']
            if gt_boxes.shape[0] == 0:
                pdb.set_trace()
            # gt_classes = gt_roidb[i]['gt_classes']
            # gt_overlaps = \
            #         model.utils.cython_bbox.bbox_overlaps(boxes.astype(np.float),
            #                                         gt_boxes.astype(np.float))
            # argmaxes = gt_overlaps.argmax(axis=1)
            # maxes = gt_overlaps.max(axis=1)
            # I = np.where(maxes > 0)[0]
            overlaps = np.zeros((num_boxes, self.num_action_classes), dtype=np.float32)
            # overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            overlaps = scipy.sparse.csr_matrix(overlaps)
            ss_roidb.append({'boxes' : boxes,
                             'gt_classes' : -np.ones((num_boxes,),
                                                      dtype=np.int32),
                             'gt_overlaps' : overlaps,
                             'flipped' : False,
                             'img_index': gt_roidb[i]['img_index']})
        return ss_roidb

    def _load_charades_annotation(self):
        """
        Load bounding box info from pkl file into data frame
        """
        annotations = pickle.load(open(os.path.join(self._data_path,'ImageSets/charadesdet_annotation_list_person.pkl'),'rb'))
        key_point_annotations = pickle.load(open(os.path.join(self._data_path, 'ImageSets/charadesdet_annotation_list_kp.pkl'), 'rb'))

        """ The annotations come from detectron detection results. The model is pre-trained on COCO
         COCO object classes classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]"""


        """
        17 Keypoints are detected: 
        ['nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle']
        """
        kp_mapping = {} # kp_mapping is a hand-crafted mapping from action class to most important kp
        with open(os.path.join(self._data_path, 'ImageSets/action2kp_mapping.txt')) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                kp_mapping[i] = int(line.rstrip().split(':')[1])
        action2obj = {}
        with open(os.path.join(self._data_path, 'ImageSets/action2obj_mapping.txt')) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                action2obj[line.split(':')[0]] = line.rstrip().split(':')[1]

        if self._image_set == 'train':
            with open(os.path.join(self._data_path, 'ImageSets/charadesdet_annotation_list.txt')) as f:
                lines = f.readlines()
        else:
            with open(os.path.join(self._data_path, 'ImageSets/charades_action_cls_test.txt')) as f:
                lines = f.readlines()
        gt_roidb = []
        new_img_index, new_index = [], []
        for ind, img_index in enumerate(self.image_index):
            rois = annotations[img_index.split("/")[1].split(".")[0]]
            person_rois = rois[1]
            inds = np.argsort(person_rois[:,4])[::-1]
            if len(inds) == 0 or person_rois[inds[0], 4] < 0.9: continue

            person_rois = person_rois[inds[0], :4].astype(np.int16)

            assert lines[ind].rstrip().split(" ")[0] == img_index
            action_clses, obj_clses = [], []
            action_flag = True
            if self._image_set == "train":
                for cls in lines[ind].rstrip().split(" ")[1].split(';')[:-1]:
                    if cls not in self._action_classes: continue
                    action_clses.append(self._action_class_to_ind[cls])
                    obj_clses.append(self._obj_class_to_ind[action2obj[cls]])
                    action_flag = False
            else:
                action_flag = False
                action_clses.append(0)
                obj_clses.append(0)
            if action_flag: continue
            new_img_index.append(img_index)
            new_index.append(ind)
            action_clses = list(set(action_clses))
            obj_clses = list(set(obj_clses))

            key_points = key_point_annotations[img_index.split("/")[1].split(".")[0]]
            kps = [k for klist in key_points for k in klist]
            kps = np.array(kps)[inds[0]]

            boxes = np.zeros((len(action_clses), 4), dtype=np.uint16)
            gt_classes = np.zeros(len(action_clses), dtype=np.int32)
            obj_classes = np.zeros(len(obj_clses), dtype=np.int32)
            overlaps = np.zeros((len(action_clses), self.num_action_classes), dtype=np.float32)
            seg_areas = np.zeros(len(action_clses), dtype=np.float32)
            ishards = np.zeros(len(action_clses), dtype=np.int32)
            key_points = np.zeros((len(action_clses), 2, 17), dtype=np.uint16)

            x1, y1, x2, y2 = person_rois
            for i, action_cls in enumerate(action_clses):
                boxes[i,:] = person_rois
                gt_classes[i] = action_cls
                overlaps[i, action_cls] = 1.0
                seg_areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1)
                key_points[i,:,:] = kps[0:2,:]
            overlaps = scipy.sparse.csr_matrix(overlaps)

            roi_dict = {'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'key_points': key_points,
                    'flipped': False,
                    'img_index': img_index}
            gt_roidb.append(roi_dict)
        self._image_index = new_img_index
        self.new_image_index = new_img_index
        self.new_index = new_index

        return gt_roidb

    def _get_charades_results_file_template(self):
        # data/charadesdevkit/results/ws_det_test_laptop.txt
        filename = 'ws_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_results_file_charades(self, all_boxes):

        for cls_ind, cls in enumerate(self._obj_classes):
            print('Writing charades {} results file'.format(cls))
            filename = self._get_charades_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind in range(len(self.roidb)):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(self.roidb[im_ind]['img_index'], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))        

if __name__ == '__main__':
    d = datasets.charades('train')
    res = d.roidb
    from IPython import embed; embed()

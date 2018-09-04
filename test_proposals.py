## Zhenheng Yang
## 07/01/2018
## ---------------------------------------------------------------
# Use the closest proposal to chosen keypoint as detection results
# Run evaluation on these detection results
# Object classes are the same as in action class names
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import numpy as np
import pickle
import cv2
import scipy.io as sio
from datasets.charades_eval import charades_eval

def do_python_eval(classes):
    annopath = 'data/charadesdevkit/ImageSets/annotations.pkl'
    imagesetfile= 'data/charadesdevkit/ImageSets/charades_det_test.txt'
    result_txt_prefix = 'res/comp12_det_test_'
    aps = []

    with open(annopath, 'rb') as f:
        try:
            recs = pickle.load(f)
        except:
            recs = pickle.load(f, encoding='bytes')

    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = result_txt_prefix+cls+'.txt'
        rec, prec, ap = charades_eval(
            filename, recs, imagesetfile, cls, ovthresh=0.5)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))

def main():

    proposal_dir = 'data/cache/selective_search_test_data_charades.mat'
    keypoint_dir = 'data/charadesdevkit/ImageSets/charadesdet_annotation_list_kp.pkl'
    action2kp_dir = 'data/charadesdevkit/ImageSets/action2kp_mapping.txt'
    action2obj_dir = 'data/charadesdevkit/ImageSets/action2obj_mapping.txt'
    test_file_list = 'data/charadesdevkit/ImageSets/charadesdet_annotation_list.txt'
    person_detection_dir = 'data/charadesdevkit/ImageSets/charadesdet_annotation_list_person.pkl'
    obj_cls_dir = 'data/charadesdevkit/ImageSets/charade_object_classes.txt'
    save_txt_prefix = 'res/comp12_det_test_'

    action2kp, action2obj = {}, {}
    with open(action2kp_dir) as f:
        lines = f.readlines()
        for line in lines:
            action2kp[line.split(" ")[0]] = int(line.rstrip().split(":")[1])
    with open(action2obj_dir) as f:
        lines = f.readlines()
        for line in lines:
            action2obj[line.split(":")[0]] = line.rstrip().split(":")[1]
    with open(obj_cls_dir) as f:
        lines = f.readlines()
        for line in lines:
            obj_clses = [line.rstrip() for line in lines]

    # do_python_eval(obj_clses)

    proposal_boxes = sio.loadmat(proposal_dir)['loadres'][0,0][1].ravel()
    keypoint_res = pickle.load(open(keypoint_dir,'rb'))
    person_bbox = pickle.load(open(person_detection_dir,'rb'))
    all_boxes = {}

    with open(test_file_list) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            img_name = line.split(' ')[0].split('.')[0].split('/')[1]
            actions = line.rstrip().split(' ')[1].split(';')[:-1]
            person_rois = person_bbox[img_name][1]
            if person_rois is None or person_rois.shape[0] == 0:
                continue
            keypoints = keypoint_res[img_name]
            kps = [k for klist in keypoints for k in klist]
            roi_index = person_rois[:,4].argmax()
            proposals = proposal_boxes[i][:, (1,0,3,2)]-1

            # img = cv2.imread('/home/zhenheng/datasets/charades/charades_rgb/'+line.split(' ')[0])
            for action in actions:
                if action not in action2kp: continue
                keypoint = kps[roi_index][0:2,[action2kp[action]]]
                object_name = action2obj[action]
                prop_centers = np.array([0.5*(proposals[:,0]+proposals[:,2]), 0.5*(proposals[:,1]+proposals[:,3])])
                center_diff = np.linalg.norm(np.abs(prop_centers - keypoint), axis=0)
                sort_ind = np.argsort(center_diff)
                det_res = proposals[sort_ind[:5],:]
                save_txt_dir = save_txt_prefix+object_name+'.txt'
                if not os.path.exists(save_txt_dir):
                    f = open(save_txt_dir, 'wt')
                else:
                    f = open(save_txt_dir, 'a')
                for res in det_res:
                    f.write(line.split(' ')[0]+' 1.0 '+' '.join([str(element) for element in res])+'\n')
                f.close()

            #     cv2.circle(img, (keypoint[0], keypoint[1]), 2, [255,0,0])
            #     for i in range(5):
            #         cv2.rectangle(img, tuple(det_res[i,0:2]), tuple(det_res[i,2:4]), (0,204,0), 2)
            #         cv2.putText(img, object_name, (det_res[i,0], det_res[i,1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imwrite('res/temp/'+img_name+'.jpg', img)
            # print('{}/{}'.format(i, len(lines)))

    do_python_eval(obj_clses)

if __name__ == '__main__':
    main()
## Zhenheng Yang
## 06/29/2018
## ------------------------------------------------------
## Utility functions for validating some random baselines
## ------------------------------------------------------

import numpy as np
import scipy.io as sio
import pickle
import torch
import sys
import pdb
sys.path.append('../../lib/')
import model.utils.cython_bbox

def calc_recall(proposals, bboxes, thresh=0.5, class_agnostic=True):

	overlaps = model.utils.cython_bbox.bbox_overlaps(proposals.astype(np.float),
													bboxes.astype(np.float))
	max_overlap = overlaps.max(axis=0)
	pos_num = np.sum(max_overlap>=thresh)
	index_negative = np.where(max_overlap<thresh)

	return pos_num, index_negative

def main():
	
	gt_bbox_dir = '../charadesdevkit/ImageSets/annotations.pkl'
	proposal_bbox_dir = '../cache/selective_search_test_data_charades.mat'
	anno_files_list = '../charadesdevkit/ImageSets/charadesdet_annotation_list.txt'
	with open(anno_files_list) as f:
		lines = f.readlines()
		anno_files = [line.split('.')[0].split('/')[-1] for line in lines]
	annotations = pickle.load(open(gt_bbox_dir))
	proposals = sio.loadmat(proposal_bbox_dir)['loadres'][0,0][1].ravel()
	num_box, num_pos_box, negative_cnt = 0, 0, 0
	negative_bbox_cnt = 0
	small_neg_bbox_cnt = 0
	threshold = 0.6

	for i, img_name in enumerate(anno_files):
		gt_boxes = np.array([line[0] for line in annotations[img_name]])
		# pdb.set_trace()
		proposal_boxes = proposals[i][:, (1,0,3,2)]-1
		# proposal_boxes = proposals[i]-1
		if gt_boxes.shape[0] == 0:
			continue
		num_pos_single, index_negative = calc_recall(proposal_boxes, gt_boxes, thresh=threshold)
		if len(index_negative[0]) != 0:
			assert gt_boxes.shape[0] == num_pos_single+len(index_negative[0])
			negative_cnt += 1
			negative_bbox_cnt += len(index_negative[0])
			cur_small_neg_bbox_cnt = np.sum(np.min([gt_boxes[index_negative[0]][:,2]-gt_boxes[index_negative[0]][:,0],gt_boxes[index_negative[0]][:,3]-gt_boxes[index_negative[0]][:,1]], axis=0) < 20)
			assert cur_small_neg_bbox_cnt<=len(index_negative[0])
			small_neg_bbox_cnt += cur_small_neg_bbox_cnt
		else:
			cur_small_neg_bbox_cnt=0

			# print(img_name)
			# print(gt_boxes[index_negative])
			# print('----------------------')

		num_box += (gt_boxes.shape[0]-cur_small_neg_bbox_cnt)
		num_pos_box += num_pos_single

	recall = float(num_pos_box)/float(num_box)
	print("Recall for threshold {} is {}".format(threshold, recall))
	print("Negative cnt: {}".format(negative_cnt))
	print("Small negative bbox out of all negative: {}/{}".format(small_neg_bbox_cnt, negative_bbox_cnt))

if __name__ == '__main__':
	main()

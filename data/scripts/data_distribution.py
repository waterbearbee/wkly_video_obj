## Zhenheng Yang
## 06/26/2018
## ---------------------------------------------------------------------------------------
## This script draws the distrbution of object location w.r.t. the most dominant key point
## The object location comes from bbox in annotations on test frames
## The key point choice is manually coded
## ---------------------------------------------------------------------------------------

import pickle
import numpy as np
import pdb
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataset_basedir = '/home/zhenheng/datasets/charades/'
action_cls_list = 'annotation/Charades_v1_classes_17obj.txt'
object_cls_list = 'charadesdet/charade_object_classes.txt'
test_file_list = 'charadesdet/charadesdet_annotation_list.txt'
bbox_annotation_file = 'charadesdet/annotations.pkl'
pose_estimation = 'charadesdet/charadesdet_annotation_list_kp.pkl'
person_detection = 'charadesdet/charadesdet_annotation_list_person.pkl'
action2kp_mapping_list = 'charadesdet/action2kp_mapping.txt'

action2obj = {}
action_names = {}
data_dist = {}
action2kp_mapping = {}
person_bbox = pickle.load(open(dataset_basedir+person_detection,'rb'))
kp_dets = pickle.load(open(dataset_basedir+pose_estimation,'rb'))
bbox_annotations = pickle.load(open(dataset_basedir+bbox_annotation_file,'rb'))

with open(dataset_basedir+action2kp_mapping_list) as f:
	lines = f.readlines()
	for line in lines:
		action2kp_mapping[line.split(' ')[0]] = int(line.rstrip().split(':')[1])
with open(dataset_basedir+object_cls_list) as f:
	lines = f.readlines()
	object_clses = [line.rstrip() for line in lines]
with open(dataset_basedir+action_cls_list) as f:
	lines = f.readlines()
	action_clses = [line.split(' ')[0] for line in lines]
	for line in lines:
		index = line.split(' ')[0]
		action_names[index] = "_".join(line.rstrip().split(' '))
		data_dist[index] = []
		for object_cls in object_clses:
			for subclass in object_cls.split('_'):
				if len(subclass)<3: continue
				if subclass in line:
					action2obj[index] = object_cls
					break

print("action2obj mapping with length: {}".format(len(action2obj)))
print(action2obj)
# with open(dataset_basedir+'charadesdet/action2obj_mapping.txt','wt') as f:
# 	for _key in action2obj:
# 		f.write(_key+':'+action2obj[_key]+'\n')




with open(dataset_basedir+test_file_list) as f:
	lines = f.readlines()
for line in lines:
	img_name = line.split(' ')[0].split('/')[1].split('.')[0]
	## delete if not visualize
	img = cv2.imread('/home/zhenheng/datasets/charades/charades_rgb/'+img_name.split('-')[0]+'/'+img_name+'.jpg')
	# pdb.set_trace()
	bboxes = bbox_annotations[img_name]
	actions = line.split(' ')[1].split(';')[:-1]
	person_rois = person_bbox[img_name][1]
	if person_rois is None or person_rois.shape[0] == 0:
		continue
	key_points = kp_dets[img_name]
	if key_points is None:
		pdb.set_trace()
	kps = [k for klist in key_points for k in klist]
	roi_index = person_rois[:,4].argmax()
	p_width, p_height = person_rois[roi_index][2]-person_rois[roi_index][0], person_rois[roi_index][3]-person_rois[roi_index][1]
	for action in actions:
		if action not in action_clses: continue
		obj_name = action2obj[action]
		for bbox in bboxes:
			if obj_name == bbox[1]:
				bbox_center = np.array([0.5*(bbox[0][0]+bbox[0][2]),0.5*(bbox[0][1]+bbox[0][3])])
				joint_center = kps[roi_index][0:2,action2kp_mapping[action]]
				center_diff = (bbox_center - joint_center)
				data_dist[action].append(center_diff)

pickle.dump(data_dist, open('data_dist.pkl','wb'))

for action in action_clses:
	if action != "c135": continue
	points = np.array(data_dist[action])
	pdb.set_trace()
	heatmap, xedges, yedges = np.histogram2d(points[:,0], points[:,1], bins=(24,24))
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	# plt.hist2d(points[:,0], points[:,1], bins=100)
	# plt.xlim([-300,300])
	# plt.ylim([-300,300])
	# plt.cla()
	# plt.plot(points[:,0], points[:,1],'bo', ms=2, alpha=.5)
	# plt.plot([0],[0], 'ro', ms=10)
	# mean_point = np.mean(points, axis=0)
	# # print(mean_point.shape)
	# plt.plot([mean_point[0]], [mean_point[1]], 'go', ms=10)
	heatmap = heatmap / heatmap.max()
	cmap = plt.cm.get_cmap('jet')
	heatmap = cmap(heatmap)
	plt.imshow(heatmap, extent=extent)
	plt.title(action_names[action])
	plt.savefig('/home/zhenheng/datasets/charades/charadesdet/data_dist/scatter/'+action_names[action].split('/')[0]+'.jpg')

# figures = os.listdir('/home/zhenheng/datasets/charades/charadesdet/data_dist/scatter/')
# figures.sort()
# whole_image = np.zeros([480*11, 640*6, 3])
# for i in range(11):
# 	row = np.zeros([480, 640*6, 3])
# 	for j in range(6):
# 		img = cv2.imread('/home/zhenheng/datasets/charades/charadesdet/data_dist/scatter/'+figures[i*6+j])
# 		row[:,640*j:640*(j+1),:] = img
# 	whole_image[480*i:480*(i+1),:,:] = row
# cv2.imwrite('/home/zhenheng/datasets/charades/charadesdet/data_dist/scatter_whl.jpg', whole_image)


				## for debugging and visualization, draw the joint center and bbox_center respectively
				
	# 			cv2.circle(img, (int(joint_center[0]), int(joint_center[1])), 2, (0,255,0))
	# 			cv2.circle(img, (int(bbox_center[0]), int(bbox_center[1])), 2, (255,0,0))
	# cv2.imwrite('/home/zhenheng/datasets/charades/charadesdet/data_dist/'+img_name+'.jpg', img)
	# print(img_name)
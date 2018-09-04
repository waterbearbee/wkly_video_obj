import torch
from torch.autograd import Variable
import pdb

def action2object_label(action_labels, action_ind_to_class, object_class_to_ind):

	batch_size = action_labels.size(0) # batch_size*5
	action2obj_mapping_dir = '/home/zhenheng/datasets/charades/charadesdet/action2obj_mapping.txt'
	action2obj_mapping = {}
	with open(action2obj_mapping_dir) as f:
		lines = f.readlines()
		for line in lines:
			action2obj_mapping[line.split(':')[0]] = line.rstrip().split(':')[1]
	action_clses  = [action_ind_to_class[_key] for _key in action_labels]
	obj_clses = [action2obj_mapping[_key] for _key in action_clses]
	object_labels = torch.LongTensor([object_class_to_ind[_key] for _key in obj_clses]).cuda() #[B*5]
	object_labels_onehot = torch.zeros(batch_size/5, len(object_class_to_ind)).cuda().scatter_(1, object_labels.view(-1,5), 1.0) # [B,17]
	# object_labels_onehot = torch.zeros(batch_size, len(object_class_to_ind)).cuda().scatter_(1, object_labels.unsqueeze(1), 1.0) #[B*5, 17]
	# object_labels_onehot = object_labels_onehot.view(-1,5,len(object_class_to_ind)).unsqueeze(1).expand(-1,700,-1,-1) #[B, 700,5,17]
	object_labels_onehot = Variable(object_labels_onehot).cuda()
	object_labels_onehot.clamp(0,1)
	return object_labels_onehot

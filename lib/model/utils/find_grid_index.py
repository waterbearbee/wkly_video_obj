## Zhenheng Yang
## 06/23/2018
## -------------------------------------------------------------------
## Utility functions for finding corresponding grid for each proposal
## grids: [B, num, 5], proposals: [B, N, 4]
## return: [B, N, k], 0<=k<num
## -------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import os.path as osp
import numpy as np
import pdb

def find_grid_index(grids, proposals):

    proposal_centers = torch.stack([0.5*(proposals[:,:,0]+proposals[:,:,2]),
                        0.5*(proposals[:,:,1]+proposals[:,:,3])], dim=2)
    num_props = proposal_centers.size(1)
    num_grids = grids.size(1) # K

    grids_lt = grids[:,:,[1,2]].unsqueeze(1).repeat(1,num_props,1,1)
    grids_lb = grids[:,:,[1,4]].unsqueeze(1).repeat(1,num_props,1,1)
    grids_rt = grids[:,:,[3,2]].unsqueeze(1).repeat(1,num_props,1,1)
    grids_rb = grids[:,:,[3,4]].unsqueeze(1).repeat(1,num_props,1,1)

    proposal_centers_mesh = proposal_centers[:,:,None,:].repeat(1,1,num_grids,1)

    rect_lt = torch.abs(proposal_centers_mesh-grids_lt)[:,:,:,0] * \
                torch.abs(proposal_centers_mesh-grids_lt)[:,:,:,1]
    rect_rt = torch.abs(proposal_centers_mesh-grids_rt)[:,:,:,0] * \
                torch.abs(proposal_centers_mesh-grids_rt)[:,:,:,1]
    rect_lb = torch.abs(proposal_centers_mesh-grids_lb)[:,:,:,0] * \
                torch.abs(proposal_centers_mesh-grids_lb)[:,:,:,1]
    rect_rb = torch.abs(proposal_centers_mesh-grids_rb)[:,:,:,0] * \
                torch.abs(proposal_centers_mesh-grids_rb)[:,:,:,1]
    rect_sum = rect_lt+rect_rt+rect_lb+rect_rb

    grids_area = (grids[:,:,3]-grids[:,:,1])*(grids[:,:,4]-grids[:,:,2])
    area_diff = rect_sum-grids_area[:,None,:].repeat(1,num_props,1)
    grid_index = (area_diff == 0)

    return grid_index
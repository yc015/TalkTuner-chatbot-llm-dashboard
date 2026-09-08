import os
import torch.nn.functional as F

import torch

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import time

tick_labels = {"gender": ["Male", "Female", "Unknown"],
               "age": ["Child", "Adolescent", "Adult", "Older Adult", "Unknown"],
               "education": ["Primary", "Secondary", "Associate", 
                             "Bachelor", "Master", "Doctoral", "Unknown"],
               "ethnics": ["Asian", "African", "White", 
                           "Hispanic", "Native Americans", "Arabs", "Unknown"],
               "socioeco": ["Low", "Middle", "High", "Unknown"],
               "marital": ["Single", "Married", "Separated", "Divorced", "Widowed"],
               "political": ["Left", "Right", "Moderate", "Unknown"]}

tic, toc = (time.time, time.time)
device = "cuda"


# N = 0
# cf_target = None


def optimize_one_inter_rep(inter_rep, layer_name, target, probe,
                           lr=1e-2, max_epoch=128, 
                           loss_func=nn.CrossEntropyLoss(), 
                           verbose=False, simplified=False, N=4, normalized=False):
                           
    
    if isinstance(target, list):
        torch_device = inter_rep.device
        target_clone = []
        for t in target:
            target_clone.append(t.clone().to(torch_device).to(torch.float))
    else:
        torch_device = inter_rep.device
        target_clone = target.clone().to(torch_device).to(torch.float)

    with torch.no_grad():
        if normalized:
            inter_rep = inter_rep + target_clone.view(1, -1) @ probe.proj[0].weight.to(torch_device) * N * 100 / rep_f().norm() 
        else:
            if isinstance(target_clone, list):
                if isinstance(N, list):
                    for i in range(len(target_clone)):
                        inter_rep += target_clone[i].view(1, -1) @ probe[i].proj[0].weight.to(torch_device) * N[i]
                else:
                    cur_input_tensor = inter_rep
                    for i in range(len(target_clone)):
                        inter_rep += target_clone[i].view(1, -1) @ probe[i].proj[0].weight.to(torch_device) * N
            else:
                inter_rep = inter_rep.clone() + target_clone.view(1, -1) @ probe.proj[0].weight.to(torch_device) * N
        # dist = torch.sqrt(torch.sum((begin_tensor - cur_input_tensor) ** 2))
    return inter_rep
    # print(probe_seg_out)
    
    # dist = torch.sqrt(torch.sum((begin_tensor - input_tensor) ** 2))
    
#     return inter_rep


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 19:42:14 2025

@author: 15834
"""

from models.model import PSP
from train_utils import DrivingData, agents_state_transition
from test_utils import testing_logging, Prediction_error
import torch
from torch import nn, optim
import numpy as np
import os
import time
from torch.utils.data import DataLoader, random_split
import logging
from torch import autograd
import yaml
import random

torch.cuda.empty_cache()
with open(r'config/config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
name = cfg["training"]["name"]
ckpt_path = eval(cfg["paths"]["ckpt_path"])
test_path = eval(cfg["paths"]["test_path"])
device = torch.device(cfg["training"]["device"])
hist_len = cfg["training"]["hist_len"]
future_len = cfg["training"]["future_len"]
batchsize = cfg["training"]["batchsize"]
training_epoch = cfg["training"]["epochs"]
seed_pool = cfg["training"]["seed_pool"]
training_epoch = cfg["training"]["epochs"]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def testing(model, test_loader, logger, device):
    model.eval()
    steps = 0
    size = len(test_loader.dataset)/batchsize
    ### testing
    for batch in test_loader:
        steps += 1
        ### data preparation
        agents = batch[0].to(device)
        maps = batch[1].to(device)
        ego_plan = batch[2].to(device)
        gt = batch[3].to(device)
        ### state iteration
        Fusion_output, Amatrix_output, Bmatrix_output, Reaction_output = model(agents, maps, ego_plan)
        agents_future = agents_state_transition(Amatrix_output, Bmatrix_output, Reaction_output, ego_plan, gt, device)
        ### evaluate
        metrics = Prediction_error(agents_future, gt, device)
        logger.log(metrics)
        print(f"test ADE in {steps}/{int(size)} batch:", round(metrics[4], 2))
#%%
if __name__ == '__main__':
    for seed in seed_pool:
        ### seed setting
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ### model loading
        model = torch.load(ckpt_path +  name + '_' + str(seed) + '_' + str(batchsize) + '_' + str(training_epoch) + '.pth')
        ### Set up dataloader
        data_set = DrivingData(eval(cfg['paths']['processed_data_dir']))
        generator = torch.Generator().manual_seed(seed)
        train_set, test_set = random_split(data_set, [0.8, 0.2], generator = generator)
        test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=True, 
                                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
        ### set up logger
        logger = testing_logging(test_path, seed, batchsize, name)
        ### testing
        testing(model, test_loader, logger, device)
        
        
        
        
        
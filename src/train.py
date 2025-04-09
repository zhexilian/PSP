# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:01:15 2025

@author: 15834
"""

from models.model import PSP
from train_utils import training_logging, DrivingData, agents_state_transition, dynamics_loss, train_metrics
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
current_path = os.getcwd()
with open(current_path + r'/config/config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
name = cfg["training"]["name"]
log_path = eval(cfg["paths"]["logging_path"])
ckpt_path = eval(cfg["paths"]["ckpt_path"])
device = torch.device(cfg["training"]["device"])
hist_len = cfg["training"]["hist_len"]
future_len = cfg["training"]["future_len"]
learning_rate = cfg["training"]["lr"]
batchsize = cfg["training"]["batchsize"]
training_epoch = cfg["training"]["epochs"]
weight_decay = cfg["training"]["weight_decay"]
optimizer_step_size = cfg["training"]["optimizer_step_size"]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_epoch(epoch, model, optimizer, scheduler, train_loader, logger, device):
    model.train()
    steps = 0
    size = len(train_loader.dataset)/batchsize
    for batch in train_loader:
        steps += 1
        ### data preparation
        agents = batch[0].to(device)
        maps = batch[1].to(device)
        ego_plan = batch[2].to(device)
        gt = batch[3].to(device)
        ### forward
        optimizer.zero_grad()
        fusion_output, Amatrix_output, Bmatrix_output, Reaction_output = model(agents, maps, ego_plan)
        ### state iteration
        agents_future = agents_state_transition(Amatrix_output, Bmatrix_output, Reaction_output, ego_plan, gt, device)
        ### loss
        loss = dynamics_loss(Reaction_output, agents_future, gt, future_len, device)
        ### backward
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()
        scheduler.step()
        ### evaluate
        planADE, planFDE = train_metrics(agents_future, gt, future_len, device)
        ### logging
        loss_print = loss.cpu().detach().numpy()
        logger.log(epoch + 1, steps, loss_print, planADE, planFDE)
        print(f"train loss in {steps}/{int(size)} batch of epoch {epoch + 1}:",float(loss_print))
        print("planADE:",planADE,"planFDE:",planFDE)
    
#%%
if __name__ == '__main__':
    seed_pool = cfg["training"]["seed_pool"]
    for seed in seed_pool:
        # seed setting
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ### Set up model
        model = PSP(hist_len, future_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_step_size, gamma=0.8)
        ### Set up dataloader
        data_set = DrivingData(eval(cfg['paths']['processed_data_dir']))
        generator = torch.Generator().manual_seed(seed)
        train_set, test_set = random_split(data_set, [0.8, 0.2], generator = generator)
        train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True, 
                                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
        ### set up logger
        logger = training_logging(log_path, seed, batchsize, name, training_epoch)
        ### training
        for epoch in range(training_epoch):
            train_epoch(epoch, model, optimizer, scheduler, train_loader, logger, device)
        
        torch.save(model, ckpt_path +  name + '_' + str(seed) + '_' + str(batchsize) + '_' + str(training_epoch) + '.pth')
        
        
        
        
    
    
    
    

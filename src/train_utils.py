# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:11:28 2024

@author: 15834
"""
import numpy as np
import torch
import csv
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
import glob
import logging

class training_logging:
    def __init__(self, log_path, seed, batchsize, name, epochs):
        self.path = log_path + name + '_' + str(seed) + '_' + str(batchsize) + '_' + str(epochs) + '.csv'
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Steps", "Train Loss", "planADE", "planFDE"])
    
    def log(self, epoch, steps, train_loss, planADE, planFDE):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, steps, train_loss, planADE, planFDE])

class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir+'*')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = torch.from_numpy(data["ego"]).unsqueeze(0)
        neighbors = torch.from_numpy(data["neighbors"])
        agents = torch.cat((ego, neighbors), dim=0)
        ego_map = torch.from_numpy(data["ego_map"])
        neighbors_map = torch.from_numpy(data["neighbors_map"])
        maps = torch.cat((ego_map, neighbors_map), dim=0)
        ego_plan = torch.from_numpy(data["ego_plan"])
        gt = torch.from_numpy(data['gt_future_states'])

        return agents, maps, ego_plan, gt
   
def dynamics_loss(Reaction_output, agents_future, gt, steps, device):
    deviation = Reaction_output[:, :, 0, :, 2:] #B*6*50*2
    scores = Reaction_output[:, :, 1] #B*6*50*4
    ground_t = gt[:, :, :,[0,3,1,2]].unsqueeze(4) #B*N*50*4*1
    ### agents loss
    agents_x_loss =  F.smooth_l1_loss(agents_future[:, :, 1:steps+1, 0], ground_t[:, :, 1:steps+1, 0])
    agents_v_loss =  F.smooth_l1_loss(agents_future[:, :, 1:steps+1, 1], ground_t[:, :, 1:steps+1, 1])
    agents_y_loss =  F.mse_loss(agents_future[:, :, 1:steps+1, 2], ground_t[:, :, 1:steps+1, 2])
    agents_fi_loss =  F.mse_loss(agents_future[:, :, 1:steps+1, 3], ground_t[:, :, 1:steps+1, 3])
    agents_loss = agents_x_loss + agents_v_loss + 10*agents_y_loss + 100*agents_fi_loss
    ### problistic loss
    Scores = scores.max(dim = -1)[0] + 1e-3
    p = torch.log(Scores)
    p_loss = F.mse_loss(-p, torch.zeros_like(p)) 
    ### deviation loss
    sigmod = deviation**2
    GMM_loss = F.mse_loss(sigmod, torch.zeros_like(sigmod))
    return agents_loss + 100 * (p_loss + GMM_loss)
    
'''Amatrix output size : B*N*(N-1)*future_lens*4*4'''
'''Bmatrix output size : B*N*(N-1)*future_lens*4*4'''
'''reaction output size : B*(N-1)*2*future lens*4'''
'''agents future size: B*N*(future lens + 1)*4*1'''
def agents_state_transition(Amatrix_output, Bmatrix_output, Reaction_output, ego_plan, gt, device):
    batchsize, agent_num, steps, feature_num = gt.shape
    reactions = Reaction_output[:, :, 0, :, :2].permute(0,1,3,2) #B*6*2*50
    plans = torch.cat((ego_plan.unsqueeze(1), reactions), dim = 1) #B*7*2*50
    agent_list = [0, 1, 2, 3, 4, 5, 6]
    ### state initialize
    agents_future = torch.zeros(batchsize, agent_num, steps, 4, 1).to(device)
    agents_future = agents_future.clone()
    agents_future[:, :, 0] = gt[:,:,0, [0,3,1,2]].unsqueeze(3)
    ### state iteration
    for t in range(steps - 1):
        for no in range(agent_num):
            new_list = [v for v in agent_list if v != no]
            Amatrix_output_temp = Amatrix_output[:,no,:,t] #B*6*4*4
            Bmatrix_output_temp = Bmatrix_output[:,no,:,t] #B*6*4*4
            gt_temp = gt[:,new_list,t][:,:,[0,3,1,2]].unsqueeze(3) #B*6*4*1
            A, B = create_AB(gt, device)
            if t == 0:
                A[:,2,3] = 0.1*gt[:,no,t,3]; B[:,3,1] = 0.1*gt[:,no,t,3]/4
                stateA = torch.cat((torch.matmul(A, gt[:,no,t,[0,3,1,2]].unsqueeze(2)).unsqueeze(1), 
                                    torch.matmul(Amatrix_output_temp, gt_temp) * 0.1), dim = 1)
            else:
                A[:,2,3] = 0.1*gt[:,no,t,3]; B[:,3,1] = 0.1*gt[:,no,t,3]/4
                stateA = torch.cat((torch.matmul(A, agents_future[:, no, t]).unsqueeze(1), 
                                    torch.matmul(Amatrix_output_temp, agents_future[:, new_list, t]-agents_future[:, new_list, t-1]) * 0.1), dim = 1)
            stateB = torch.cat((torch.matmul(B, plans[:, no, :, t].unsqueeze(2)).unsqueeze(1),
                                torch.matmul(Bmatrix_output_temp, plans[:, new_list, :, t].unsqueeze(3)) * 0.1), dim = 1)
            agents_future = agents_future.clone()
            agents_future[:, no, t + 1] = torch.sum(stateA, dim = 1) + torch.sum(stateB, dim = 1)
    
    return agents_future

def find_inf(H):
    mask = torch.abs(H) > 1e3
    H[mask] = torch.sqrt(H[mask])
    if torch.isinf(H).any():
        inf_indices = torch.isinf(H).nonzero(as_tuple=True)
        # 对于找到的每个inf位置，将其替换为上一行同列同深度的元素
        for i in range(len(inf_indices[0])):
            x, y, z, w = inf_indices[0][i], inf_indices[1][i], inf_indices[2][i], inf_indices[3][i]
            if y > 0:  # 确保不是在第一行，否则没有上一行
                H[x, y, z, w] = H[x, y-1, z, w]
    if torch.isnan(H).any():
        nan_indices = torch.isnan(H).nonzero(as_tuple=True)
        # 对于找到的每个inf位置，将其替换为上一行同列同深度的元素
        for i in range(len(nan_indices[0])):
            x, y, z, w = nan_indices[0][i], nan_indices[1][i], nan_indices[2][i], nan_indices[3][i]
            if y > 0:  # 确保不是在第一行，否则没有上一行
                H[x, y, z, w] = H[x, y-1, z, w]
    return H

           
def create_AB(gt,device):
    '''diagonal matrix'''
    A = torch.zeros(gt.shape[0],4,4).to(device)
    B = torch.zeros(gt.shape[0],4,2).to(device)
    A[:,...] = torch.FloatTensor(np.array([[1,0.1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    B[:,...] = torch.FloatTensor(np.array([[0,0],[0.1,0],[0,0],[0,0]]))  
    return A, B      

def train_metrics(agents_future, gt, steps, device):
    ground_t = gt[:, :, :,[0,3,1,2]].unsqueeze(4) #B*N*50*4*1
    plan_distance = torch.norm(agents_future[:, :, 1:steps+1, [0,2]] - ground_t[:, :, 1:steps+1, [0,2]], dim=-1)
    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(torch.norm(agents_future[:, :, -1, [0,2]] - ground_t[:, :, -1, [0,2]], dim=-1))
    
    return plannerADE.item(), plannerFDE.item(),
      

def loss_plot(loss):
    epoch = np.linspace(1,len(loss[2:]),len(loss[2:]))
    fig = plt.figure(figsize=(24,15))
    plt.plot(epoch,loss[2:],"r-",lw=3,label="train_loss")
    plt.ylim([0,800])
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.xlabel("epoch",fontsize=36)
    plt.ylabel("Loss",fontsize=36)
    plt.legend(fontsize=24)
    plt.grid()    
    plt.show()
    plt.close()



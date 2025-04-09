# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 19:54:36 2025

@author: 15834
"""
import csv
import torch

class testing_logging:
    def __init__(self, test_path, seed, batchsize, name):
        self.path = test_path + name + '_' + str(seed) + '_' + str(batchsize) + '.csv'
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ADE_1s", "ADE_2s", "ADE_3s", "ADE_4s", "ADE_5s",
                             "RMSE_1s", "RMSE_2s", "RMSE_3s", "RMSE_4s", "RMSE_5s", "FDE",
                             "HE_1s", "HE_2s", "HE_3s", "HE_4s", "HE_5s",
                             "SE_1s", "SE_2s", "SE_3s", "SE_4s", "SE_5s"])
    
    def log(self, metrics):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metrics)

def Prediction_error(agents_future, gt, device):
    ground_t = gt[:, 1:, :,[0,3,1,2]] #B*N*50*4
    agents_future = agents_future[:, 1:].squeeze(4)
    ADE_1s = torch.mean(torch.norm(agents_future[:, :, 1:11, [0,2]] - ground_t[:, :, 1:11, [0,2]], p=2, dim=-1)).item()
    RMSE_1s = torch.sqrt(torch.mean((agents_future[:, :, 1:11, [0,2]] - ground_t[:, :, 1:11, [0,2]]).pow(2).sum(dim = -1))).item()
    ADE_2s = torch.mean(torch.norm(agents_future[:, :, 1:21, [0,2]] - ground_t[:, :, 1:21, [0,2]], p=2, dim=-1)).item()
    RMSE_2s = torch.sqrt(torch.mean((agents_future[:, :, 1:21, [0,2]] - ground_t[:, :, 1:21, [0,2]]).pow(2).sum(dim = -1))).item()
    ADE_3s = torch.mean(torch.norm(agents_future[:, :, 1:31, [0,2]] - ground_t[:, :, 1:31, [0,2]], p=2, dim=-1)).item()
    RMSE_3s = torch.sqrt(torch.mean((agents_future[:, :, 1:31, [0,2]] - ground_t[:, :, 1:31, [0,2]]).pow(2).sum(dim = -1))).item()
    ADE_4s = torch.mean(torch.norm(agents_future[:, :, 1:41, [0,2]] - ground_t[:, :, 1:41, [0,2]], p=2, dim=-1)).item()
    RMSE_4s = torch.sqrt(torch.mean((agents_future[:, :, 1:41, [0,2]] - ground_t[:, :, 1:41, [0,2]]).pow(2).sum(dim = -1))).item()
    ADE_5s = torch.mean(torch.norm(agents_future[:, :, 1:51, [0,2]] - ground_t[:, :, 1:51, [0,2]], p=2, dim=-1)).item()
    RMSE_5s = torch.sqrt(torch.mean((agents_future[:, :, 1:51, [0,2]] - ground_t[:, :, 1:51, [0,2]]).pow(2).sum(dim = -1))).item()
    FDE = torch.mean(torch.norm(agents_future[:, :, 50, [0,2]] - ground_t[:, :, 50, [0,2]], p=2, dim=-1)).item()
    HE_1s = torch.mean(torch.norm(agents_future[:, :, 1:11, [3]] - ground_t[:, :, 1:11, [3]], p=2, dim=-1)).item()
    HE_2s = torch.mean(torch.norm(agents_future[:, :, 1:21, [3]] - ground_t[:, :, 1:21, [3]], p=2, dim=-1)).item()
    HE_3s = torch.mean(torch.norm(agents_future[:, :, 1:31, [3]] - ground_t[:, :, 1:31, [3]], p=2, dim=-1)).item()
    HE_4s = torch.mean(torch.norm(agents_future[:, :, 1:41, [3]] - ground_t[:, :, 1:41, [3]], p=2, dim=-1)).item()
    HE_5s = torch.mean(torch.norm(agents_future[:, :, 1:51, [3]] - ground_t[:, :, 1:51, [3]], p=2, dim=-1)).item()
    SE_1s = torch.mean(torch.norm(agents_future[:, :, 1:11, [1]] - ground_t[:, :, 1:11, [1]], p=2, dim=-1)).item()
    SE_2s = torch.mean(torch.norm(agents_future[:, :, 1:21, [1]] - ground_t[:, :, 1:21, [1]], p=2, dim=-1)).item()
    SE_3s = torch.mean(torch.norm(agents_future[:, :, 1:31, [1]] - ground_t[:, :, 1:31, [1]], p=2, dim=-1)).item()
    SE_4s = torch.mean(torch.norm(agents_future[:, :, 1:41, [1]] - ground_t[:, :, 1:41, [1]], p=2, dim=-1)).item()
    SE_5s = torch.mean(torch.norm(agents_future[:, :, 1:51, [1]] - ground_t[:, :, 1:51, [1]], p=2, dim=-1)).item()
    return [ADE_1s, ADE_2s, ADE_3s, ADE_4s, ADE_5s,
            RMSE_1s, RMSE_2s, RMSE_3s, RMSE_4s, RMSE_5s, FDE,
            HE_1s, HE_2s, HE_3s, HE_4s, HE_5s,
            SE_1s, SE_2s, SE_3s, SE_4s, SE_5s]
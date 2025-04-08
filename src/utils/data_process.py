# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:55:19 2024

@author: Frank Yan
"""

import numpy as np
import pandas as pd
import math
import warnings
from data_utils import agent_norm, map_norm
from scenario_visualization import scenario_viz
import gc
import matplotlib.pyplot as plt
import os
import random
import shutil
import yaml
import glob
warnings.filterwarnings('ignore')

class DataProcess(object):
    def __init__(self, data1):
        self.num_neighbors = 6
        self.hist_len = 40
        self.future_len = 50
        self.map_length = 100
        self.max_neighbor_distance = 50
        self.data = data1
        self.data = self.data[self.data["Lane_ID"]<=6].reset_index(inplace=False, drop = True)
        self.map_dict = self.build_map() 
    
    def build_map(self):
        '''
        map dictionary
        '''
        map_dict = {}
        ## lane1 - lane6
        for lane_num in range(1,7):
            lane = self.data[self.data["Lane_ID"]==lane_num].reset_index(inplace = False)
            center_x = round(np.mean(lane["Local_X"]),2)
            min_y, max_y = math.floor(lane["Local_Y"].min()), math.ceil(lane["Local_Y"].max())
            if lane_num <= 5:
                map_dict[lane_num] = np.array([[i, center_x, 0, 0, 30] for i in range(min_y,max_y,1)])
            else:
                map_dict[lane_num] = np.array([[i, center_x, 0, 1, 30] for i in range(min_y,max_y,1)])
        return map_dict
    
    def lane_map_feature(self, lane_id, pos):
        '''
        get_agents_map_feature
        '''
        lanes = {1,2,3,4,5,6}
        if lane_id not in lanes:
            return np.zeros((self.map_length, 5))
        points = self.map_dict[lane_id]
        low, high = 0, len(points)
        while low < high:
            mid = (low + high) // 2
            if points[mid][0] > pos:
                high = mid
            else:
                low = mid + 1
        i = high
        if self.map_dict[lane_id].shape[0] - i > self.map_length:
            return self.map_dict[lane_id][i:i+self.map_length,:]
        else:
            map_feature = np.zeros((self.map_length,5))
            map_feature[0:self.map_dict[lane_id].shape[0] - i,:] = self.map_dict[lane_id][i:,:]
            return map_feature
    
    def ego_process(self, group, timestep):
        start_idx = timestep + 1 - self.hist_len
        selected = group.loc[start_idx:timestep, [
                            'Local_Y', 'Local_X', 'Heading', 
                            'velocity_y', 'velocity_x', 
                            'acc_y', 'acc_x']]
        ego_states = selected.to_numpy()
        
        # get current state
        self.current_xyh = ego_states[-1, :3]
        self.laneID = group.loc[timestep, ['Lane_ID']][-1]
        #size:(40,7)
        return ego_states.astype(np.float32)
    
    def find_surrounding_vehicles(self, neighbors):
        ego_x, ego_y = self.current_xyh[:2]
        direction_categories = {'FV': None, 'RV': None,
                                'LFV': None, 'LRV': None,
                                'RFV': None, 'RRV': None}
        min_distance = {key: float('inf') for key in direction_categories}
        for vehicle_id, (x, y, lane) in neighbors.items():
            dx = x - ego_x
            dy = y - ego_y
            dl = int(lane - self.laneID)
            distance = math.hypot(dx, dy)
            if dl == 0:
                if dx > 0: direction = 'FV'
                else: direction = 'RV'
            elif dl == -1:
                if dx > 0: direction = 'LFV'
                else: direction = 'LRV'
            elif dl == 1:
                if dx > 0: direction = 'RFV'
                else: direction = 'RRV'
            else: 
                continue
            # 更新最近车辆
            if distance < min_distance[direction]:
                min_distance[direction] = distance
                direction_categories[direction] = vehicle_id
        return direction_categories
    
    def neighbors_process(self,sdc_id, group, timestep):
        start_time = group.loc[timestep+1-self.hist_len].Global_Time
        current_time = group.loc[timestep].Global_Time
        neighbors_states = np.zeros(shape=(self.num_neighbors, self.hist_len, 7))
        neighbors = {}
        
        # search for nearby agents
        search_data = self.data[self.data["Global_Time"]==current_time].reset_index(inplace=False, drop = True)
        non_ego_mask = search_data["Vehicle_ID"] != sdc_id
        neighbors_data = search_data.loc[non_ego_mask, ["Vehicle_ID", "Local_Y", "Local_X", "Lane_ID"]]
        if not neighbors_data.empty:
            neighbor_ids = neighbors_data["Vehicle_ID"].to_numpy()
            neighbor_coords = neighbors_data[["Local_Y", "Local_X", "Lane_ID"]].to_numpy()
            neighbors.update(zip(neighbor_ids, neighbor_coords))
        # filter the agents by distance
        sorted_neighbors = dict([obj for obj in neighbors.items() if np.linalg.norm(obj[1][:2] - self.current_xyh[:2])<=self.max_neighbor_distance])
        # identify different vehicles
        self.surrounding_neighbors = self.find_surrounding_vehicles(neighbors)
        # add neighbor agents into the array
        for num, item in enumerate(self.surrounding_neighbors.items()):
            neighbor_id = item[1]
            if neighbor_id == None: continue
            neighbor_data = self.data[self.data["Vehicle_ID"]==neighbor_id].reset_index(inplace=False, drop=True) 
            neighbor_data = neighbor_data[neighbor_data["Global_Time"]<=current_time].reset_index(inplace=False, drop=True)
            if len(neighbor_data) < self.hist_len: 
                return np.zeros(shape=(self.num_neighbors, self.hist_len, 7)),self.surrounding_neighbors
            selected = neighbor_data.loc[len(neighbor_data) - self.hist_len:len(neighbor_data), [
                                'Local_Y', 'Local_X', 'Heading', 
                                'velocity_y', 'velocity_x', 
                                'acc_y', 'acc_x']]
            neighbors_states[num] = selected.to_numpy()
                              
        # size = (6, 40, 7) [FV, RV, LFV, LRV, RFV, RRV]
        return neighbors_states.astype(np.float32), self.surrounding_neighbors
     
    def ego_map_process(self,sdc_id, group, timestep):
        ego_map_feature = np.zeros((1, 3, self.map_length, 5))
        lane_id = group.loc[timestep].Lane_ID
        ego_pos = group.loc[timestep].Local_Y
        #lane features
        lane_left = self.lane_map_feature(lane_id-1,ego_pos) 
        lane_ego = self.lane_map_feature(lane_id, ego_pos) 
        lane_right = self.lane_map_feature(lane_id+1, ego_pos) 
        ego_map_feature[0,...] = np.stack((lane_left,lane_ego,lane_right),axis=0)
        # size = (1, 3, 100, 5)
        return ego_map_feature.astype(np.float32)
    
    def neighbor_map_process(self, group, timestep):
        current_time = group.loc[timestep].Global_Time
        neighbor_map_feature = np.zeros((self.num_neighbors,3,self.map_length,5))
        for i, veh in enumerate(self.surrounding_neighbors.items()):
            neighbor_id = veh[1]
            if neighbor_id == None: continue
            lane_id = self.data[self.data["Vehicle_ID"]==neighbor_id][self.data["Global_Time"]==current_time].reset_index(inplace=False)["Lane_ID"][0]
            pos = self.data[self.data["Vehicle_ID"]==neighbor_id][self.data["Global_Time"]==current_time].reset_index(inplace=False)["Local_Y"][0]
            #lane features
            lane_left = self.lane_map_feature(lane_id-1,pos) 
            lane_ego = self.lane_map_feature(lane_id, pos) 
            lane_right = self.lane_map_feature(lane_id+1, pos)
            neighbor_map_feature[i,...] = np.stack((lane_left,lane_ego,lane_right),axis=0)
        # size = (self.neighbors_num, 3, 50, 5)
        return neighbor_map_feature.astype(np.float32)
    
    def groundtruth_process(self,sdc_id, group, timestep):
        ground_truth = np.zeros(shape=(1+self.num_neighbors, self.future_len+1, 7))
        end_time = group.loc[timestep+self.future_len].Global_Time
        current_time = group.loc[timestep].Global_Time
        ## ego ground_truth
        selected = group.loc[timestep:timestep+self.future_len, [
                            'Local_Y', 'Local_X', 'Heading', 
                            'velocity_y', 'velocity_x', 
                            'acc_y', 'acc_x']]
        ground_truth[0] = selected.to_numpy()
        # neighbors ground truth data
        for i, veh in enumerate(self.surrounding_neighbors.items()):
            neighbor_id = veh[1]
            if neighbor_id == None: continue
            track_states = self.data[self.data["Vehicle_ID"]==neighbor_id][self.data["Global_Time"]>=current_time].reset_index(inplace=False, drop=True)   
            if len(track_states) < self.future_len+1:
                return np.zeros(shape=(1+self.num_neighbors, self.future_len+1, 7))
            selected = track_states.loc[0:self.future_len, [
                                'Local_Y', 'Local_X', 'Heading', 
                                'velocity_y', 'velocity_x', 
                                'acc_y', 'acc_x']]
            ground_truth[i + 1] = selected.to_numpy()
        self.gt = ground_truth
        #size = (1+self.neighbor_nums, futrue_lens + 1, 7)
        return ground_truth.astype(np.float32)
    
    def ego_plan(self, group, timestep):
        '''compute ego's control input in open loop'''
        end_idx = timestep + self.future_len - 1
        selected = group.loc[timestep:end_idx, ['acc_y', 'wheelangle']]
        self.ego_command = selected.to_numpy().T
        return self.ego_command.astype(np.float32)
    
    def normalize_data(self, ego, neighbors, ego_map, neighbors_map, ground_truth):
        # get the center and heading (local view)
        center, angle = self.current_xyh[:2], self.current_xyh[2]
        # normalize agents
        ego = agent_norm(ego, center, 0)
        ground_truth[0] = agent_norm(ground_truth[0], center, 0) 
        
        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                neighbors[i] = agent_norm(neighbors[i], center, 0, impute=True)
            if ground_truth[i+1, -1, 0] != 0:
                ground_truth[i+1] = agent_norm(ground_truth[i+1], center, 0)  
        # normalize map
        ego_map = map_norm(ego_map, center, 0)
        for i in range(neighbors.shape[0]):
            if neighbors_map[i, 1, -1, 0] != 0:
                neighbors_map[i] = map_norm(neighbors_map[i].reshape(1,3,self.map_length,5), center, 0)
        return ego.astype(np.float32), neighbors.astype(np.float32), ground_truth.astype(np.float32), ego_map.astype(np.float32), neighbors_map.astype(np.float32)
    
    def nan_check(self, ego, neighbors, ego_map, neighbors_map, ground_truth, ego_plan):
        if np.isnan(ego).any() or np.isnan(ego_map).any() or np.isnan(neighbors).any() or np.isnan(neighbors_map).any() or np.isnan(ground_truth).any() or np.isnan(ego_plan).any():
            return True
        else:
            return False
    
    def yaw_rate_check(self,agents):
        heading_change = np.array([agents[i,2] - agents[i-1,2] for i in range(1,len(agents))])
        if np.any(np.abs(heading_change) > 0.15):
            return True
        else:
            return False
    
    def data_check(self, timestep, ego, neighbors, ego_map, neighbors_map, ground_truth, ego_plan):
        # check yaw_rate
        if np.any(np.abs(ego_plan[0])>10):
            print(timestep,"acceleration out of bounds")
            return False
        if np.any(np.abs(ego_plan[1])>1):
            print(timestep,"angle out of bounds")
            return False
        # check nan
        if self.nan_check(ego, neighbors, ego_map, neighbors_map, ground_truth, ego_plan):
            print(timestep, "nan exists")
            return False
        return True
        
    def process_data(self, save_path, viz = False):
        self.IDgroup_data = self.data.groupby(["Vehicle_ID","Total_Frames"])
        process_num = 0
        for name, group in self.IDgroup_data:
            sdc_id = name[0] #int
            frames = name[1]
            time_len = len(group) #int
            
            group = group.reset_index(inplace=False, drop = True)
            process_num += 1
            # start collect data
            for timestep in range(self.hist_len + 1, time_len - self.future_len - 1, 10):
                # ego data
                ego = self.ego_process(group, timestep)
                # neighbors data
                neighbors, _ = self.neighbors_process(sdc_id, group, timestep)
                if np.all(neighbors == 0):
                    print("historical data is not enough")
                    continue
                # map data
                ego_map = self.ego_map_process(sdc_id, group, timestep)
                neighbors_map = self.neighbor_map_process(group, timestep)
                # ground truth data
                ground_truth = self.groundtruth_process(sdc_id, group, timestep)
                if np.all(ground_truth == 0):
                    print("future data is not enough")
                    continue
                # ego plan
                ego_plan = self.ego_plan(group, timestep)
                # data check
                if self.data_check(timestep, ego, neighbors, ego_map, neighbors_map, ground_truth, ego_plan) == False:
                    continue
                # normalize
                ego, neighbors, ground_truth, ego_map, neighbors_map = self.normalize_data(ego, neighbors, ego_map, neighbors_map, ground_truth)
                if viz:
                    scenario_viz(ego, neighbors, ego_map, ground_truth, sdc_id, frames, timestep)
                # save data
                filename = f"{save_path}/{sdc_id}_{frames}_{timestep}.npz"
                np.savez(filename, ego=ego, neighbors=neighbors, ego_map=ego_map, neighbors_map=neighbors_map, 
                         gt_future_states=ground_truth, ego_plan = ego_plan)
                print(f"{sdc_id}_{frames}_{timestep} has done! Progress:{process_num}/{len(self.IDgroup_data)}")
                del ego, neighbors, ground_truth, ego_map, neighbors_map, ego_plan
            gc.collect()                 
#%%    
if __name__ == '__main__':
    current_path = os.getcwd()
    # load config
    with open(current_path + r'/config/config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    # get NGSIM original data
    original_data_path = eval(cfg['paths']['origin_data_dir'])
    data = pd.read_csv(glob.glob(original_data_path + '*')[0])
    # data process
    ngsim = DataProcess(data)
    ngsim.process_data(eval(cfg['paths']['processed_data_dir']))
    
    

    


    

    
    

    
    
    

    
    
    
    
    
    
    
    
    
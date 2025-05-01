# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:07:37 2024

@author: 15834
"""

import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
#from train_utils import agents_state_transition, DrivingData, train_metrics, dynamics_loss, find_inf
from torch.utils.data import DataLoader
import yaml
import glob
import random

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class SpatioAttention(nn.Module):
    def __init__(self, embed_dim = 128, num_heads = 8):
        super().__init__()
        # time attention
        self.time_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, embed_dim))
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, inputs, x):
        # [batch_size, num_agents, time_steps, features]
        batch_size, num_agents, time_steps, features = x.shape
        # time attention
        inputs_time = inputs.view(batch_size * num_agents, time_steps, 7)
        mask = torch.eq(inputs_time[:, :, 0], 0)
        all_masked = mask.all(dim=1)
        mask[all_masked, -1] = False
        x_time = x.view(batch_size * num_agents, time_steps, features)
        x_time, _ = self.time_attn(x_time, x_time, x_time, key_padding_mask = mask)  # [batch*agents, time, features]
        x_time = x_time.view(batch_size, num_agents, time_steps, features)
        output = self.ln(self.ffn(x_time) + x)
        return output

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim = 128, num_heads = 8):
        super().__init__()
        # agent attetion
        self.agent_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, embed_dim))
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, inputs, x):
        # [batch_size, num_agents, time_steps, features]
        batch_size, num_agents, time_steps, features = x.shape
        # agent attention
        inputs = inputs.permute(0, 2, 1, 3).contiguous()
        inputs_agents = inputs.view(batch_size * time_steps, num_agents, 7)
        mask = torch.eq(inputs_agents[:, :, 0], 0)
        all_masked = mask.all(dim=1)
        mask[all_masked, -1] = False
        x_agent = x.permute(0, 2, 1, 3)
        x_agent = x_agent.reshape(batch_size * time_steps, num_agents, features)
        x_agent, _ = self.agent_attn(x_agent, x_agent, x_agent, key_padding_mask = mask)  # [batch*time, agents, features]
        x_agent = x_agent.view(batch_size, time_steps, num_agents, features)
        x_agent = x_agent.permute(0, 2, 1, 3)
        output = self.ln(self.ffn(x_agent) + x)
        return output

class SelfTransformer(nn.Module):
    def __init__(self, dim = 128):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim, 8, 0.1, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, dim))
        self.ln = nn.LayerNorm(dim)
    def forward(self, input, mask=None):
        attention_output, _ = self.self_attention(input, input, input, key_padding_mask=mask)
        output = self.ln(self.ffn(attention_output) + input)
        return output

class CrossTransformer(nn.Module):
    def __init__(self, dim = 128):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, 8, 0.1, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, dim))
        self.ln = nn.LayerNorm(dim)
    def forward(self, query, key, mask=None):
        value = key
        if mask != None: mask[:, 0] = False
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.ln(self.ffn(attention_output) + query)
        return output
    
'''input size: B*N*T*7 means B batchsize N agents T history time steps 7 features'''
class AgentEncoder(nn.Module):

    def __init__(self, hist_len):
        super(AgentEncoder, self).__init__()
        self.encode = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 128))
        self.position = PositionalEncoding(d_model=128, max_len=hist_len)
        self.spatio_attn1 = SpatioAttention()
        self.temporal_attn1 = TemporalAttention()

    def forward(self, inputs):
        inputs_1 = self.position(self.encode(inputs))
        spatio = self.spatio_attn1(inputs, inputs_1)
        temporal = self.temporal_attn1(inputs, spatio)
        spatio = self.spatio_attn1(inputs, temporal)
        temporal = self.temporal_attn1(inputs, spatio)
        return temporal

'''input size: B*N*L*P*5 means B batchsize N agents number L lanes P future waypoints 5 features'''
class MapEncoder(nn.Module):
    def __init__(self):
        super(MapEncoder, self).__init__()
        self.waypoint = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 32))

    def forward(self, inputs):
        output = self.waypoint(inputs)
        return output

'''input size: B*N*T*128 means B batchsize N agents T history time steps 128 embedding features'''
class Agent2Agent(nn.Module):
    def __init__(self, histlen = 40, features = 128):
        super(Agent2Agent, self).__init__()
        self.ffn = nn.Sequential(nn.Linear(10*features, 640), nn.ReLU(), nn.Dropout(0.1), 
                                 nn.Linear(640, 320), nn.ReLU(), nn.Dropout(0.1),nn.Linear(320, 128))
        self.interaction_1 = SelfTransformer()
        self.interaction_2 = SelfTransformer()

    def forward(self, agents, inputs):
        mask = torch.eq(agents[:, :, 0, 0], 0)
        # sparse sample
        inputs_sampled = inputs[:,:,0::4,:].contiguous()
        batch_size, num_agents, time_steps, features = inputs_sampled.shape
        agents_v2v = self.ffn(inputs_sampled.view(batch_size, num_agents,time_steps*features)) #B*N*128
        output = self.interaction_1(agents_v2v, mask=mask)
        output = self.interaction_2(output, mask=mask)
        return output
    
'''agent input size: B*N*T*128 means B batchsize N agents T history time steps 128 embedding features'''
'''map input size: B*N*L*P*128 means B batchsize N agents number L lanes P future waypoints 128 embedding features'''
class Agent2Map(nn.Module):
    def __init__(self, num_lane = 3, map_len = 100, hist_len = 40):
        super(Agent2Map, self).__init__()
        self.hist_len = hist_len
        self.ffn1 = nn.Sequential(nn.Linear(num_lane*10*32, 640), nn.ReLU(), nn.Dropout(0.1), 
                                 nn.Linear(640, 320), nn.ReLU(), nn.Dropout(0.1), nn.Linear(320, 128))
        self.ffn2 = nn.Sequential(nn.Linear(1280, 640), nn.ReLU(), nn.Dropout(0.1), 
                                 nn.Linear(640, 320), nn.ReLU(), nn.Dropout(0.1),nn.Linear(320, 128))
        self.interaction_1 = CrossTransformer()
        self.interaction_2 = SelfTransformer()

    def forward(self, agents, maps, inputs):
        mask = torch.eq(maps[:, :, 1, 0, 0], 0)
        inputs_sampled = inputs[:, :, :, 0::10, :].contiguous()
        agents_sampled = agents[:,:,0::4,:].contiguous()
        batch_size, num_agents, num_lanes, num_points, features = inputs_sampled.shape
        inputs_sampled = self.ffn1(inputs_sampled.view(batch_size, num_agents, num_lanes*num_points*features))  #B*N*128
        agents_sampled = self.ffn2(agents_sampled.view(batch_size, num_agents, agents_sampled.shape[2]*agents_sampled.shape[3]))
        output = self.interaction_1(agents_sampled, inputs_sampled, mask)
        output = self.interaction_2(output, mask)
        return output
'''v2v_feature: B*N*features B-Batchsize N-agnts'''
'''v2m_feature: B*N*features B-Batchsize N-agnts'''
class Fusion(nn.Module):
    def __init__(self, feature_dim = 256):
        super(Fusion, self).__init__()
        self.interaction_1 = SelfTransformer(feature_dim)
        self.interaction_2 = SelfTransformer(feature_dim)
        self.act = nn.ReLU()
    def forward(self, agents, v2v_feature, v2m_feature):
        mask = torch.eq(agents[:, :, 0, 0], 0)
        fusion = torch.cat((v2v_feature, v2m_feature), dim=2)
        fusion = self.interaction_1(fusion, mask)
        fusion = self.act(self.interaction_2(fusion, mask))
        return fusion

'''fusion feature B*N*256'''
'''output size : B*(N-1)*future_lens*4*4'''
class AMatrixDecoder(nn.Module):
    def __init__(self,steps):
        super(AMatrixDecoder, self).__init__()
        self.steps = steps
        self.attention = CrossTransformer(256)
        self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(256, steps*16), nn.LayerNorm(steps*16)) for _ in range(6)])
    def forward(self, inputs, num):
        ego_inputs = inputs[:, num, :].unsqueeze(1)
        _ = torch.arange(inputs.shape[1]) != num
        agent_inputs = inputs[:, _, :]
        attention_output = self.attention(ego_inputs, agent_inputs)
        attention_output = attention_output.reshape(attention_output.shape[0],-1)
        outputs = [ffn(attention_output).view(-1, self.steps, 4, 4).unsqueeze(1) 
                  for ffn in self.ffns]
        return torch.cat(outputs, dim=1)

'''fusion feature B*N*256'''
'''output size : B*(N-1)*future_lens*4*2'''
class BMatrixDecoder(nn.Module):
    def __init__(self,steps):
        super(BMatrixDecoder, self).__init__()
        self.steps = steps
        self.attention = CrossTransformer(256)
        self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(256, steps*8), nn.LayerNorm(steps*8)) for _ in range(6)])
    def forward(self, inputs, num):
        ego_inputs = inputs[:, num, :].unsqueeze(1)
        _ = torch.arange(inputs.shape[1]) != num
        agent_inputs = inputs[:, _, :]
        attention_output = self.attention(ego_inputs, agent_inputs)
        attention_output = attention_output.reshape(attention_output.shape[0],-1)
        outputs = [ffn(attention_output).view(-1, self.steps, 4, 2).unsqueeze(1) 
                  for ffn in self.ffns]
        return torch.cat(outputs, dim=1)
'''fusion input: B*N*256; ego plan input: B*2*future lens; agents input: B*N*hist lens*128'''
'''res output: B*2*future lens*4'''
class GMMPredictor(nn.Module):
    def __init__(self, future_len, modes):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.modes = modes
        self.encode = nn.Sequential(nn.Linear(100, 128), nn.ReLU(), nn.Linear(128, 256), nn.LayerNorm(256))
        self.ffn = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.LayerNorm(256))
        self.position = PositionalEncoding(d_model=256, max_len = future_len)
        self.attention1 = CrossTransformer(256)
        self.attention2 = CrossTransformer(256)
        self.gaussian = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 16), nn.LayerNorm(16))
        self.score = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1),
                                   nn.Linear(32, self.modes * 1),nn.Softmax(dim=2))
    def forward(self, inputs, ego_plan, agents_output, num):
        #encoding ego_plan
        ego_plan_process = ego_plan.view(-1, ego_plan.shape[1]*ego_plan.shape[2])
        plans =self.position(self.encode(ego_plan_process).unsqueeze(1)).squeeze(1)
        attention_output = self.attention1(plans, inputs)
        attention_output = self.attention2(attention_output, self.ffn(agents_output[:,num]))
        gussian_output = self.gaussian(attention_output)
        B, M = inputs.shape[0], self.modes
        res = gussian_output.view(B, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(attention_output)
        #selected Gussian component
        score_index = torch.max(score, dim=2)[1]
        batch_idx = torch.arange(B)[:, None, None]
        modal_idx = score_index[:, :, None]
        pos_idx = torch.arange(50)[None, :, None]
        prediction = res[batch_idx, modal_idx, pos_idx, :].squeeze(2)
        output = torch.cat((prediction.unsqueeze(1), score.unsqueeze(1)), dim = 1)
        return output

'''agent input size: B*N*T*7 means B batchsize N agents T history time steps 7 features'''
'''map input size: B*N*L*P*5 means B batchsize N agents number L lanes P future waypoints 5 features'''
'''fusion output: B*N*features B-Batchsize N-agnts'''
class Encoder(nn.Module):
    def __init__(self, histlen):
        super(Encoder, self).__init__()
        self.histlen = histlen
        self.agent_encoder = AgentEncoder(histlen)
        self.map_encoder = MapEncoder()
        self.v2v_encoder = Agent2Agent()
        self.v2m_encoder = Agent2Map()
        self.fusion_encoder = Fusion()
    def forward(self, agents, maps):
        agents_output = self.agent_encoder(agents)
        maps_output = self.map_encoder(maps)
        v2v_output = self.v2v_encoder(agents, agents_output)
        v2m_output = self.v2m_encoder(agents_output, maps, maps_output)
        fusion_output = self.fusion_encoder(agents, v2v_output, v2m_output)
        return agents_output, fusion_output     

'''Amatrix output size : B*N*(N-1)*future_lens*4*4'''
'''Bmatrix output size : B*N*(N-1)*future_lens*4*4'''
'''reaction output size : B*(N-1)*2*future lens*4'''
class Decoder(nn.Module):
    def __init__(self,steps):
        super(Decoder, self).__init__()
        self.steps = steps
        self.num_agents = 7
        self.Amatrix_decoder = AMatrixDecoder(self.steps)
        self.Bmatrix_decoder = BMatrixDecoder(self.steps)
        self.Reaction_decoder = GMMPredictor(self.steps, 4)
    
    def forward(self,inputs, ego_plan, agents_output):
        Amatrix_output = torch.cat([self.Amatrix_decoder(inputs, num).unsqueeze(1) for num in range(self.num_agents)], dim = 1)
        Bmatrix_output = torch.cat([self.Bmatrix_decoder(inputs, num).unsqueeze(1) for num in range(self.num_agents)], dim = 1)
        Reaction_output = torch.cat([self.Reaction_decoder(inputs, ego_plan, agents_output, num).unsqueeze(1) for num in range(1, self.num_agents)], dim = 1)
        return Amatrix_output, Bmatrix_output, Reaction_output

class PSP(nn.Module):
    def __init__(self, histlen, steps):
        super(PSP, self).__init__()
        self.encoder = Encoder(histlen)
        self.decoder = Decoder(steps)
    def forward(self, agents, maps, ego_plan):
        agents_output, fusion_output = self.encoder(agents, maps)
        Amatrix_output, Bmatrix_output, Reaction_output = self.decoder(fusion_output, ego_plan, agents_output)
        return fusion_output, Amatrix_output, Bmatrix_output, Reaction_output
#%%
if __name__ == '__main__':
    with open(r'..\config\config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    data_list = glob.glob(eval(cfg['paths']['processed_data_dir']) + '*')
    num = random.randint(0, 67500)
    data = np.load(data_list[0])
    '''import feature'''
    ego = torch.from_numpy(data["ego"]).unsqueeze(0).unsqueeze(1)
    neighbors = torch.from_numpy(data["neighbors"]).unsqueeze(0)
    agents = torch.cat((ego, neighbors), dim=1)
    ego_map = torch.from_numpy(data["ego_map"]).unsqueeze(0)
    neighbors_map = torch.from_numpy(data["neighbors_map"]).unsqueeze(0)
    maps = torch.cat((ego_map, neighbors_map), dim=1)
    ego_plan = torch.from_numpy(data["ego_plan"]).unsqueeze(0)
    print("input NaN:", torch.isnan(agents).any(), torch.isnan(maps).any(), torch.isnan(ego_plan).any())
    '''network test'''
    model = PSP(40, 50)
    Amatrix_output, Bmatrix_output, Reaction_output = model(agents, maps, ego_plan)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params}")
    # gt = data['gt_future_states'].reshape((1,4,11,7)); 

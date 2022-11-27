# Based on RelationNet's implement by ynie in Total3DUnderstanding

import torch
import torch.nn as nn
from configs.data_config import Relation_Config

rel_cfg = Relation_Config()
class RelationNet(nn.Module):
    # In this implement, ynie use the original appearance feature to replace the V_feature
    def __init__(self):
        super(RelationNet, self).__init__()
        self.df = 2048
        self.dg = rel_cfg.d_g
        self.dk = rel_cfg.d_k
        self.Nr = rel_cfg.Nr
        self.K = nn.Linear(self.df, self.dk * self.Nr)
        self.Q = nn.Linear(self.df, self.dk * self.Nr)
        self.G = nn.Linear(self.dg, 1 * self.Nr)
        # self.threshold = nn.ReLU()
        self.threshold = nn.Threshold(1e-6, 1e-6) # In paper Relation Networks, it use Relu
        self.softmax = nn.Softmax(dim = 1)
        self.scale_layer = nn.Conv1d(1,1,1)

    def forward(self, a_feature, g_feature, split, pair_counts):
        '''
        a_feature: PatchSize (sum of Several N_i) x 2048, appearance feature
        g_feature: Number of pairs(sum of several N_i^2) x 64, geometric feature
        split: Shows how N_i construct the whole patch, e.g. [[0,5],[5,8]] means N_0 = 5, N_1 = 3
        pair_counts: Shows which image(?) the geometric feature (between a pair) belongs to. eg [0, 49, 113] 0-49(49), 49-113(64)
        '''

        g_weight = self.G(g_feature)
        g_weight = self.threshold(g_weight) # N^2 x Nr
        g_weight = g_weight.transpose(0,1) # Nr x N^2

        k_featrue = self.K(a_feature) # N x (Nr x dk)
        q_feature = self.Q(a_feature) # N x (Nr x dk)

        k_featrue = k_featrue.view(-1, self.Nr, self.dk) # N x Nr x dk
        q_feature = q_feature.view(-1, self.Nr, self.dk) # N x Nr x dk
        
        k_featrue = k_featrue.transpose(0, 1) # Nr x N x dk
        q_feature = q_feature.transpose(0, 1) # Nr x N x dk

        v_feature = a_feature.view(a_feature.size(0), self.Nr, -1) #N x Nr x (2048/Nr)
        v_feature = v_feature.transpose(0, 1) # Nr x N x (2048/Nr)

        sqrt_dk = torch.sqrt(torch.tensor(self.dk))
        
        r_features = []

        for idx, interval in enumerate(split):
            sample_k_feature = k_featrue[:, interval[0]:interval[1], :] # N is the number of objects in a single Image Now!
            sample_q_feature = k_featrue[:, interval[0]:interval[1], :]

            sample_g_weights = g_weight[:, pair_counts[idx]:pair_counts[idx+1]] # Nr x (N_i)^2
            sample_g_weights = sample_g_weights.view(self.Nr, interval[1]-interval[0], interval[1]-interval[0]) # Nr x N x N

            sample_a_weight = torch.bmm(sample_k_feature, sample_q_feature.transpose(1,2)) # Nr x N x N 
            sample_a_weight = torch.div(sample_a_weight, sqrt_dk)

            sample_r_weight = self.softmax(torch.log(sample_g_weights) + sample_a_weight) # Nr x N x N 

            sample_v_feature = v_feature[:,interval[0]:interval[1],:] # Nr x N x 2048/Nr We use the different part of a_feature with different weight.

            sample_r_feature = torch.bmm(sample_v_feature.transpose(1,2) ,sample_r_weight) # Nr x 2048/Nr x N

            sample_r_feature = sample_r_feature.view(-1, sample_r_feature.size(2)) # 2048 x N
            sample_r_feature = sample_r_feature.transpose(0, 1) # N x 2048

            r_features.append(sample_r_feature)

        r_features = torch.cat(r_features, dim = 0) # (sum N) x 2048
        r_features = self.scale_layer(r_features.unsqueeze(1)).squeeze(1) #N x 1 x 2048 -> Conv1 -> N x 2048

        return r_features
    
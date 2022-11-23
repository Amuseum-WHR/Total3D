'''
Objection Detection Net
Based on Total3DUnderstanding by ynie
'''

import torch
import torch.nn as nn
import model.utils.resnet as resnet
import torch.utils.model_zoo as model_zoo
from model.ODN.Relation_Net import RelationNet
from configs.data_config import NYU40CLASSES
class ODN(nn.Module):
    def __init__(self, cfg):
        super(ODN, self).__init__()
        bin = cfg.dataset_config.bins
        self.ORI_BIN_NUM = len(bin['ori_bin'])
        self.CENTER_BIN_NUM = len(bin['centroid_bin'])

        self.resnet = resnet.resnet34(pretrained = False)

        self.relnet = RelationNet()

        Linear_input_size = 2048 + len(NYU40CLASSES) # Depend on the dataset

        # For Projection offset (2D)
        self.fc1 = nn.Linear(Linear_input_size, 128)
        self.fc2 = nn.Linear(128, 2)
        
        # For distance to Camara Center, which is used to calculate Centroid of 3D Box 
        self.fc3 = nn.Linear(Linear_input_size, 128)
        self.fc4 = nn.Linear(128, 2 * self.CENTER_BIN_NUM)

        # For the size of 3D Box
        self.fc5 = nn.Linear(Linear_input_size, 128)
        self.fc6 = nn.Linear(128, 3)

        # For orientation of 3D box (1 dimension)
        self.fc7 = nn.Linear(Linear_input_size, 128)
        self.fc8 = nn.Linear(128, 2 * self.ORI_BIN_NUM)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)
        # Load Pretrained ResNet Model
        pretrained_dict = model_zoo.load_url(resnet.model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

        # initialize weights (I don't understand why it is necessary)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()
    def forward(self, x, Geometry_features, split, pair_counts, target):
        '''
            x: Patch x Channel x Hight x Width
            Geometry_feature: sum of pairs x 64
            split: e.g. [[0,5],[5,8]]
            pair_counts: e.g. [0,49,113]
            target: Patch x num of class: target[i]: object i should belongs to class[i], where targets[i][class[i]] =1 (?)
        '''
        a_featrues = self.resnet(x)
        a_features = a_featrues.view(a_featrues.size(0),-1) # N x 2048
        r_features = self.relnet(a_featrues, Geometry_features, split, pair_counts) # N x 2048
        
        a_r_featrues = torch.add(a_features, r_features) # N x 2048
        a_r_featrues = torch.cat([a_r_featrues, target], dim = 1) # N x (2048 + class)

        # Use Fc to predict all we want
        offset = self.fc1(a_r_featrues)
        offset = self.relu(offset)
        offset = self.dropout(offset)
        offset = self.fc2(a_r_featrues)

        distance = self.fc3(a_r_featrues)
        distance = self.relu(distance)
        distance = self.dropout(distance)
        distance = self.fc4(a_r_featrues)
        distance = distance.view(-1, self.CENTER_BIN_NUM, 2)
        distance_cls = distance[:,:,0]
        distance_reg = distance[:,:,0]

        size = self.fc4(a_r_featrues)
        size = self.relu(size)
        size = self.dropout(size)
        size = self.fc6(size)

        orientation = self.fc7(a_r_featrues)
        orientation = self.relu(orientation)
        orientation = self.dropout(orientation)
        orientation = self.fc8(orientation)
        orientation = orientation.view(-1, self.ORI_BIN_NUM, 2)
        orientation_cls = orientation[:,:,0]
        orientation_reg = orientation[:,:,1]

        return size, orientation_reg, orientation_cls, distance_reg, distance_cls, offset
import json
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from scipy.spatial import cKDTree


root = ""    #根目录
pix3d = "data/pix3d/demo_data/"    #Pix3d数据目录
mod = 'train'

HEIGHT_PATCH = 256
WIDTH_PATCH = 256
pix3d_n_classes = 9
neighbors = 30

non_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
        
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((280, 280)),
    transforms.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PixDataset(Dataset):
    def __init__(self,root_path = root, pix3d_path = pix3d, mode = 'train', transform = trans, device = "cuda"):
        '''
        self.root_path:    Get the data_path
        self.sunrgbd_path = pix3d_path:    Get the Sunrgbd dataset path
        self.file_idx = []:    Get the file index, 但这里由于其train dataset 和 test dataset 选取数据非常混乱，因此直接load .json文件
        '''
        self.root_path = root_path    
        self.pix3d_path = pix3d_path
        self.transform = transform
        self.file_idx = []
        for i in range(7):
            self.file_idx.append('{}.pkl'.format(i))
        self.device = device

    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx):
        '''
        dict_keys: 'class_id', 'gt_3dpoints', 'sample_id', 'img'
        
        In the keys, also dicts:
        
        class_id:    图像类别
        sample_id:    图像id，图像是数据集中第几个，那么id就是几
        Gt_3dpoints:   为10000个三维坐标数据，是一个二维列表[[x,y,z],…,[x',y',z']].
        img:    图像,三维列表 H*W*channel, 但每张图片大小并不相同.
                    
        '''
        data_pkl = pickle.load(open(self.root_path + self.pix3d_path + self.file_idx[idx],'rb'))
        data_img = data_pkl['img']
        gt_points = data_pkl['gt_3dpoints']
        data_class = data_pkl['class_id']
        
        cls_codes = torch.zeros(pix3d_n_classes)
        cls_codes[data_class] = 1
        
        '''
        
        dists是到最近邻居的距离，indices是每个邻居的索引，两者大小均为 10000*30
        离每一个3d_points最近的是它自己，因此dists[index][0] = 0.0, 因此计算密度是用dists[index][1]
        '''
        tree = cKDTree(gt_points)    #提高空间索引速度
        dists, indices = tree.query(gt_points, k=neighbors)    
        densities = np.array([max(dists[point_set, 1]) ** 2 for point_set in indices])
        
        '''
        Output:
            为每一个数据生成一个Sample,Sample中keys(): sequence_id, img, cls mesh_poings, densities;
            sequence_id: 数据在数据集中的序号，
            img: shape [batch_size, 3, 256, 256]
            cls: shape [batch_size, pix3d_n_classes] 独热向量, pix3d_n_classes = 9.
            mesh_points 
            
        '''

        gt_points = torch.FloatTensor(gt_points)
        densities = torch.FloatTensor(densities)
        
        if self.transform is None:
            sample = {'sequence_id':data_pkl['sample_id'],
                      'img':self.transform(image),
                      'cls':cls_codes,
                      'mesh_points':gt_points,
                      'densities': densities}
            return sample
        else:
            data_raw = data_img
            data_img = self.transform(data_img)
            sample = {'sequence_id':data_pkl['sample_id'],
                      'img':data_img,
                      'cls':cls_codes,
                      'mesh_points':gt_points,
                      'densities': densities}
            return sample

if __name__ == 'main':
    dataset2 = PixDataset(transform=trans, root_path=root, pix3d_path=pix3d, mode=mod)
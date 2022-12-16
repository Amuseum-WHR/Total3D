import math
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import collections

class Relation_Config(object):
    def __init__(self):
        self.d_g = 64
        self.d_k = 64
        self.Nr = 16

HEIGHT_PATCH = 256
WIDTH_PATCH = 256
NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
rel_cfg = Relation_Config()
d_model = int(rel_cfg.d_g/4)
default_collate = torch.utils.data.dataloader.default_collate

root = '../'    #根目录
sunrgbd = 'data/sunrgbd/sunrgbd_train_test_data/'    #Pix3d数据目录
mod = 'train'
div = "cuda"

non_trans = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trans = transforms.Compose([
    transforms.Resize((280, 280)),
    transforms.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

pil2tensor = transforms.ToTensor()

class SunDataset_2DBB(Dataset):
    def __init__(self,root_path = root,RGBD_path = sunrgbd,mode = 'train',transform = None, device = "cuda"):
        '''
        self.root_path:    Get the data_path
        self.sunrgbd_path = RGBD_path:    Get the Sunrgbd dataset path
        self.file_idx = []:    Get the file index, train: idx 5051 to 10335, test: idx 1 to 5050
        '''
        self.root_path = root_path    
        self.sunrgbd_path = RGBD_path    
        self.file_idx = []
        self.device = device
        self.transform = transform

        if (mode == 'train'):
            self.transform = trans
            with open(self.root_path + '/data/sunrgbd/splits/train.json') as json_file:
                self.file_idx = json.load(json_file)
        elif (mode == 'test'):
            self.transform = non_trans
            with open(self.root_path + '/data/sunrgbd/splits/test.json') as json_file:
                self.file_idx = json.load(json_file) 
    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self,idx):
        '''
        As input, 
        dict_keys: 'rgb_img', 'depth_map', 'boxes', 'camera', 'layout', 'sequence_id'
        
        In the keys, also dicts:
        
        'rgb_img':  shape(H,W,3)
        'depth_map':  shape(H,W)
        'layout':  'ori_cls', 'ori_reg', 'centroid_reg', 'coeffs_reg', 'bdb3D'
        
        'boxes':    'ori_cls', 'ori_reg', 'size_reg', 'bdb3D', 'bdb2D_from_3D', 
                    'bdb2D_pos', 'centroid_cls', 'centroid_reg', 'delta_2D', 'size_cls', 'mask'
        'Sequence_id': 序列号，但不等于在数据集中的序号
                    
        '''

        data_pkl = pickle.load(open(self.root_path+ self.file_idx[idx][1:],'rb'))
        image = Image.fromarray(data_pkl['rgb_img'])
        depth = Image.fromarray(data_pkl['depth_map'])
        camera = data_pkl['camera']
        boxes = data_pkl['boxes']

        # build relational geometric features for each object
        n_objects = boxes['bdb2D_pos'].shape[0]

        # g_feature: n_objects x n_objects x 4
        # Note that g_feature is not symmetric,
        # g_feature[m, n] is the feature of object m contributes to object n.
        # TODO: think about it, do we need to involve the geometric feature from each object itself?
        g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                      ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                      math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                      math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                     for id1, loc1 in enumerate(boxes['bdb2D_pos'])
                     for id2, loc2 in enumerate(boxes['bdb2D_pos'])]

        locs = [num for loc in g_feature for num in loc]

        pe = torch.zeros(len(locs), d_model)
        position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        boxes['g_feature'] = pe.view(n_objects * n_objects, rel_cfg.d_g)

        labels = boxes['size_cls']

        # encode class
        cls_codes = torch.zeros([len(boxes['size_cls']), len(NYU40CLASSES)])
        cls_codes[range(len(boxes['size_cls'])), boxes['size_cls']] = 1
        boxes['size_cls'] = cls_codes
       
        layout = data_pkl['layout']

        #image.crop()切割图片，按照左上右下拒图片左上边界的距离将图片截取出来
        patch = []
        for bdb in boxes['bdb2D_pos']:
            img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            img = trans(img)
            patch.append(img)

        boxes['patch'] = torch.stack(patch)
        image = non_trans(image)
        
        #问题出在 boxes['mask'] 上，为什么会出现这个问题呢？
        dic1 = {}
        for key in boxes:
            if key != 'mask':
                dic1[key] = boxes[key]
            if key == 'mask':
                tmp = boxes[key]
                arr = []
                for i in range(len(tmp)):
                    if boxes[key][i] is None:
                        arr.append({})
                    else:
                        arr.append(boxes[key][i])
                dic1[key] = arr
        
        bbox = [[dic1['bdb2D_pos'][i][1], dic1['bdb2D_pos'][i][0], dic1['bdb2D_pos'][i][3], dic1['bdb2D_pos'][i][2]]  for i in range(n_objects)]
        
        return (np.float32(data_pkl['rgb_img'].transpose((2,0,1))), np.float32(bbox), labels)

def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem

def collate_fn(batch):
    """
    Data collater.
    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batches: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {}
    # iterate over keys
    for key in batch[0]:
        if key == 'boxes_batch':
            collated_batch[key] = dict()
            for subkey in batch[0][key]:
                if subkey == 'mask':
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    tensor_batch = torch.cat(list_of_tensor)
                collated_batch[key][subkey] = tensor_batch
        elif key == 'depth':
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            collated_batch[key] = default_collate([elem[key] for elem in batch])

    interval_list = [elem['boxes_batch']['patch'].shape[0] for elem in batch]
    collated_batch['obj_split'] = torch.tensor([[sum(interval_list[:i]), sum(interval_list[:i+1])] for i in range(len(interval_list))])

    return collated_batch

def Sunrgbd_dataloader(mode='train'):
    dataloader = DataLoader(dataset=SunDataset_2DBB(mode = mode),
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    '''
        dataloader: return index and a dict
        dict_keys: 'image', 'depth', 'boxes_batch', 'camera', 'layout', 'sequence_id'
        'image': torch.Size([1, 3, 256, 256])
        'depth': torch.Size([1, 256, 256])
    '''
    dataset2 = SunDataset_2DBB(transform=trans, root_path=root, RGBD_path=sunrgbd, mode=mod,device=div)
    dataloader = Sunrgbd_dataloader(mode=mod)

import torch
import torch.nn as nn

from model.LEN.LEN_model import PoseNet as LEN
from configs import data_config
from model import loss
from dataset.SunrgbdDataloader import SunDataset, collate_fn
from easydict import EasyDict
from torch.utils.data import DataLoader
import argparse
from model.utils.libs import to_dict_tensor

from torch.utils.tensorboard import SummaryWriter

import os
import datetime
import time
from tqdm import *

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bottleneck_size", type = int, default = 1024, help='dim_out_patch')
    parser.add_argument("--number_points", type = int, default = 2562)
    parser.add_argument("--subnetworks", type = int, default = 2, help='num of tnn subnetworks')
    parser.add_argument("--face_samples", type = int, default = 1, help='num of face_samples')
    parser.add_argument("--num_classes", type = int, default = 9, help='num of classes of dataset')
    parser.add_argument("--threshold", type = float, default = 0.001, help='threshold of tnn network')
    parser.add_argument("--factor", type = float, default = 0.5, help='factor of tnn network')

    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--betas", type = list, default = [0.9, 0.999])
    parser.add_argument("--eps", type = float, default = 1e-08)
    parser.add_argument("--weight_decay", type = float, default = 1e-04)
    parser.add_argument("--batch_size", type = int, default = 32, help = 'Batch Size' )
    parser.add_argument("--nepoch", type = float, default = 500, help = 'the total training epochs')

    parser.add_argument("--model_path", type = str, default = "out", help = 'path of saved model')
    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "len", help = 'name of this training process')
    
    parser.add_argument("--demo", action="store_true", default = False, help = 'demo or not')
    parser.add_argument("--demo_path", type = str, default = 'demo')
    parser.add_argument("--check_freq", type = int, default = 5, help = 'The frequency of print loss in screen.')
    parser.add_argument("--save_freq", type = int, default = 10, help = 'The frequency of saving a model.')
    parser.add_argument("--cuda", type =str, default = "cuda:7", help = 'Which GPU to use for training.')
    parser.add_argument("--cuda_num", type =int, default = 7, help = 'Which GPU to use for training.')
    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt

class Trainer():
    def __init__(self, cfg, net, optimizer, Loss, device = None):
        self.net = net
        self.cfg = cfg
        self.optimizer = optimizer
        self.device = device
        self.Loss = Loss
        self.bins_tensor = to_dict_tensor(cfg.bins, if_cuda=True if device != 'cpu' else False)
    def to_device(self, data):
        '''
        Change the data of SRGBD into the inputs we want to use.
        And to device.
        '''
        device = self.device
        image = data['image'].to(device)
        pitch_reg = data['camera']['pitch_reg'].float().to(device)
        pitch_cls = data['camera']['pitch_cls'].long().to(device)
        roll_reg = data['camera']['roll_reg'].float().to(device)
        roll_cls = data['camera']['roll_cls'].long().to(device)
        lo_ori_reg = data['layout']['ori_reg'].float().to(device)
        lo_ori_cls = data['layout']['ori_cls'].long().to(device)
        lo_centroid = data['layout']['centroid_reg'].float().to(device)
        lo_coeffs = data['layout']['coeffs_reg'].float().to(device)
        lo_bdb3D = data['layout']['bdb3D'].float().to(device)

        layout_input = {'image':image, 'pitch_reg':pitch_reg, 'pitch_cls':pitch_cls, 'roll_reg':roll_reg,
                        'roll_cls':roll_cls, 'lo_ori_reg':lo_ori_reg, 'lo_ori_cls':lo_ori_cls, 'lo_centroid':lo_centroid,
                        'lo_coeffs':lo_coeffs, 'lo_bdb3D':lo_bdb3D}

        return layout_input
        
    def train_step(self, data):
        self.optimizer.zero_grad()
        data = self.to_device(data)
        
        est_data = self.net(data['image'])
        loss, _ = self.Loss(est_data, data, self.bins_tensor)
        loss['total'].backward()
        self.optimizer.step()
        return loss
    
    # def eval_step(self, data):
    #     est_data = self.net(data)
    #     loss = self.Loss(est_data, data)
    #     return loss, est_data

if __name__ == "__main__":
    opt = parser()
    if torch.cuda.is_available():
        opt.device = torch.device(opt.cuda)
        torch.cuda.set_device(opt.cuda_num) 
    else:
        opt.device = torch.device("cpu")
    
    # init_log
    log_path = os.path.join(opt.log_path, opt.name, 'summary')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(opt.log_path, opt.name, 'loss_log.txt')

    def log_start_train():
            with open(log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    Writer = SummaryWriter()
    dataset = SunDataset(root_path='.', device= opt.device)
    cfg = data_config.Config('sunrgbd')
    net = LEN(cfg).to(opt.device)
    Train_loader = DataLoader(dataset, batch_size= opt.batch_size, collate_fn=collate_fn, shuffle = False)

    optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr, betas = opt.betas, eps = opt.eps, weight_decay=opt.weight_decay)    
    trainer = Trainer(cfg, net, optimizer, loss.PoseLoss(), opt.device)

    epochs = opt.nepoch
    net.train = True
    for epoch in range(epochs):
        loop = tqdm(enumerate(Train_loader), total=len(Train_loader))
        for idx, gt_data in loop:
            steploss = trainer.train_step(gt_data)
            for key, value in steploss.items():
                Writer.add_scalar('train/loss_' + key, scalar_value=value, global_step=idx + epoch * len(Train_loader))
            message = '( epoch: %d, ) ' % (epoch)
            message += '( step: %d, ) ' % (idx)
            message += '%s: %.5f' % ("loss_train_total", steploss['total'])
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss = steploss['total'])

        if epoch % opt.check_freq == 0:
            print('epoch {} loss: {:.4f}'.format(epoch, steploss['total']))

        if (epoch % opt.save_freq ==0 ):
            print("saving nat...")            
            if not os.path.exists(os.path.join(opt.model_path, opt.name)):
                os.makedirs(os.path.join(opt.model_path, opt.name))
            model_path = opt.model_path + '/'+ opt.name + '/model_epoch{}.pth'.format(epoch)
            torch.save(net.state_dict(), model_path)
            print("network saved")
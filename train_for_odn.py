import torch
import torch.nn as nn

from model.ODN import ODN_model
from configs import data_config
from model import loss
from dataset.SunrgbdDataloader import SunDataset, collate_fn
from easydict import EasyDict
from torch.utils.data import DataLoader
import argparse

from torch.utils.tensorboard import SummaryWriter

import os
import datetime
import time

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bottleneck_size", type = int, default = 1024, help='dim_out_patch')
    parser.add_argument("--number_points", type = int, default = 2562)
    parser.add_argument("--subnetworks", type = int, default = 2, help='num of tnn subnetworks')
    parser.add_argument("--face_samples", type = int, default = 1, help='num of face_samples')
    parser.add_argument("--num_classes", type = int, default = 9, help='num of classes of dataset')
    parser.add_argument("--threshold", type = float, default = 0.001, help='threshold of tnn network')
    parser.add_argument("--factor", type = float, default = 0.5, help='factor of tnn network')

    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--betas", type = list, default = [0.9, 0.999])
    parser.add_argument("--eps", type = float, default = 1e-08)
    parser.add_argument("--weight_decay", type = float, default = 1e-04)

    parser.add_argument("--nepoch", type = float, default = 500, help = 'the total training epochs')

    parser.add_argument("--model_path", type = str, default = "out", help = 'path of saved model')
    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "test_code", help = 'name of this training process')
    
    parser.add_argument("--demo", type = bool, default = True, help = 'demo or not')
    parser.add_argument("--demo_path", type = str, default = 'demo')

    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt

class Trainer():
    def __init__(self, net, optimizer, Loss, device = None):
        self.net = net
        self.optimizer = optimizer
        self.device = device
        self.Loss = Loss
    
    def to_device(self, data):
        '''
        Change the data of SRGBD into the inputs we want to use.
        And to device.
        '''

        device = self.device
        patch = data['boxes']['patch'].to(device)
        g_features = data['boxes']['g_feature'].float().to(device)
        size_reg = data['boxes']['size_reg'].float().to(device)
        size_cls = data['boxes']['size_cls'].float().to(device)
        ori_reg = data['boxes']['ori_reg'].float().to(device)
        ori_cls = data['boxes']['ori_cls'].float().to(device)
        centroid_reg = data['boxes']['centroid_reg'].float().to(device)
        centroid_cls = data['boxes']['centroid_cls'].float().to(device)
        offset_2D = data['boxes']['delta_2D'].float().to(device)
        split = data['obj_split']
        rel_pair_count = torch.cat([torch.tensor([0]), torch.cumsum(
            torch.pow(data['obj_split'][:,1]- data['obj_split'][:,0],2),dim = 0)],dim = 0)
        object_input = {'patch': patch, 'g_features': g_features, 'size_reg': size_reg, 'size_cls': size_cls,
                        'ori_reg': ori_reg, 'ori_cls': ori_cls, 'centroid_reg': centroid_reg, 'centroid_cls': centroid_cls,
                        'offset_2D': offset_2D, 'split': split, 'rel_pair_counts': rel_pair_count}
        return object_input
    def train_step(self, data):
        self.optimizer.zero_grad()
        data = self.to_device(data)
        est_data = self.net(data)
        loss = self.Loss(est_data, data)
        loss['total'].backward()
        self.optimizer.step()
        return loss
    
    # def eval_step(self, data):
        # est_data = self.net(data)
        # loss = self.Loss(est_data, data)
        # return loss

if __name__ == "main":
    opt = parser()
    if torch.cuda.is_available():
        opt.device = torch.device("cuda:1")
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
    dataset = SunDataset()
    net = ODN_model(data_config)
    dataset = SunDataset(device = opt.device)
    Train_loader = DataLoader(dataset, batch_size= 2, collate_fn=collate_fn, shuffle = True)

    optimizer = torch.optim.Adam(net.parameters(), lr = opt.lr, betas = opt.betas, eps = opt.eps, weight_decay=opt.weight_decay)    
    trainer = Trainer(net, optimizer, loss.Detection_Loss, opt.device)

    epochs = opt.nepoch
    net.train = True
    for epoch in range(epochs):
        for idx, gt_data in enumerate(Train_loader):
            loss = trainer.train_step(gt_data)
            Writer.add_scalar('train/loss', scalar_value=loss, global_step=idx + epochs * len(Train_loader))
            message = '( epoch: %d, ) ' % (epoch)
            message += '( step: %d, ) ' % (idx)
            message += '%s: %.5f' % ("loss_train_total", loss.item())
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)
                
        if epoch % opt.check_freq == 0:
            print('epoch {} loss: {:.4f}'.format(epoch, loss.item()))

        if (epoch % opt.save_freq ==0 ):
            print("saving nat...")            
            if not os.path.exists(os.path.join(opt.model_path, opt.name)):
                os.makedirs(os.path.join(opt.model_path, opt.name))
            model_path = opt.model_path + '/'+ opt.name + '/model_epoch{}.pth'.format(epoch)
            torch.save(net.state_dict(), model_path)
            print("network saved")
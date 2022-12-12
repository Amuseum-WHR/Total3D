from trainer import Trainer

import time
import datetime
import argparse
import os

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from dataset.SunrgbdDataloader import SunDataset, collate_fn

from torch.utils.tensorboard import SummaryWriter

from tqdm import *


def parser():
    parser = argparse.ArgumentParser()

    # for mgn
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

    parser.add_argument("--mgn_load_path", type = str, default = "out/mgn_pretrain_model.pth", help = 'path of saved mgn model')
    parser.add_argument("--len_load_path", type = str, default = "out/len_pretrain_model.pth", help = 'path of saved odn model')
    parser.add_argument("--odn_load_path", type = str, default = "out/odn_pretrain_model.pth", help = 'path of saved len model')
    parser.add_argument("--t3d_load_path", type = str, default = "", help = 'path of saved t3d model')

    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "total3d", help = 'name of this training process')
    
    parser.add_argument("--demo", action="store_true", default = False, help = 'demo or not')
    parser.add_argument("--demo_path", type = str, default = 'demo')
    parser.add_argument("--check_freq", type = int, default = 5, help = 'The frequency of print loss in screen.')
    parser.add_argument("--save_freq", type = int, default = 10, help = 'The frequency of saving a model.')
    parser.add_argument("--cuda", type =str, default = "cuda:0", help = 'Which GPU to use for training.')
    parser.add_argument("--cuda_num", type =int, default = 0, help = 'Which GPU to use for training.')
    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt


if __name__ == "__main__":
    opt = parser()
    if torch.cuda.is_available():
        opt.device = torch.device(opt.cuda)
        torch.cuda.set_device(opt.cuda_num) 
    else:
        opt.device = torch.device("cpu")
    
    Writer = SummaryWriter()
    dataset = SunDataset(root_path='.', device= opt.device)
    Train_loader = DataLoader(dataset, batch_size= opt.batch_size, collate_fn=collate_fn, shuffle = False)
    trainer = Trainer(opt, device=opt.device)
    trainer.model.to(opt.device)

    log_path = os.path.join(opt.log_path, opt.name, 'summary')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(opt.log_path, opt.name, 'loss_log.txt')

    def log_start_train():
            with open(log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    epochs = opt.nepoch
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
            trainer.save_net(model_path)
            print("network saved")
import argparse
import os
import datetime
import time
import numpy as np
from easydict import EasyDict
import model.MGN.MGN_model as model
from model.loss import SVRLoss
import dataset.Pix3dDataloader as pixed_loader
import dataset.pix3d_loader_single as demo_loader
import torch
from torch.utils.data import DataLoader

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bottleneck_size", type = int, default = 1024, help='dim_out_patch')
    parser.add_argument("--number_points", type = int, default = 2562)
    parser.add_argument("--subnetworks", type = int, default = 2, help='num of tnn subnetworks')
    parser.add_argument("--face_samples", type = int, default = 1, help='num of face_samples')
    parser.add_argument("--num_classes", type = int, default = 9, help='num of classes of dataset')
    parser.add_argument("--threshold", type = float, default = 0.2, help='threshold of tnn network')
    parser.add_argument("--factor", type = float, default = 0.5, help='factor of tnn network')

    parser.add_argument("--lr", type = float, default = 2e-4) 
    parser.add_argument("--betas", type = list, default = [0.9, 0.999])
    parser.add_argument("--eps", type = float, default = 1e-08)
    parser.add_argument("--weight_decay", type = float, default = 1e-04)
    parser.add_argument("--batch_size", type = int, default = 16)

    parser.add_argument("--nepoch", type = float, default = 20, help = 'the total training epochs')

    parser.add_argument("--model_path", type = str, default = "out", help = 'path of saved model')
    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "mgn", help = 'name of this training process')
    parser.add_argument("--save_freq", type = int, default = 1)
    parser.add_argument("--log_freq", type = int, default = 8)
    parser.add_argument("--check_freq", type = int, default = 15)
    parser.add_argument("--pretrain", type = bool, default = True)
    parser.add_argument("--load_epoch", type = int, default = None)
    parser.add_argument("--freeze", type = bool, default = True)
    
    parser.add_argument("--demo", type = bool, default = False, help = 'demo or not')
    parser.add_argument("--demo_path", type = str, default = 'demo')
    parser.add_argument("--demo_num", type = int, default = 3)

    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt

opt = parser()
if torch.cuda.is_available() and opt.demo == False:
    opt.device = torch.device(f"cuda:0")
else:
    opt.device = torch.device(f"cpu")

print("current device: " + str(opt.device))
# opt.load_path = opt.model_path + '/'+ opt.name + '/model_epoch{}.pth'.format(opt.load_epoch)

# init_log
log_path = os.path.join(opt.log_path, opt.name, 'summary')
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_name = os.path.join(opt.log_path, opt.name, 'loss_log.txt')

def log_start_train():
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

def mgn_loss(est_data, gt_data, opt=opt):
    svrloss = SVRLoss()(est_data, gt_data, subnetworks = opt.subnetworks, face_sampling_rate = opt.face_samples)
    total_loss = sum(svrloss.values())
    for key, item in svrloss.items():
            svrloss[key] = item.item()
    return {'total':total_loss, **svrloss}

net = model.EncoderDecoder(opt)
dataset_train = pixed_loader.PixDataset()
dataset_demo = demo_loader.PixDataset()
# net.decoder.load_state_dict(torch.load(opt.load_path))

def load(filename):
    checkpoint = torch.load(filename)
    model_dict = net.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint.items():
        module = v
        for key, value in module.items():
            change_key = key[27:]
            if change_key[0:7] != 'encoder':
                change_key = 'decoder.' + change_key
            pretrained_dict[change_key] = value
            # print(change_key)
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

if opt.pretrain == True:
    opt.load_path = opt.model_path + '/pretrain' + '/meshnet_model.pth'
    load(opt.load_path)
elif opt.load_epoch != None:
    try:
        net.load_state_dict(torch.load(opt.load_path))
        print("Loading pretrianed model " + opt.load_path)
    except:
        print("Can not find pretrianed model " + opt.load_path)
if opt.freeze == True:
    net.freeze_tmn()
    print("ee modules frozen")

if opt.demo == False:
    pix_loader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)
else:
    pix_loader = DataLoader(dataset_demo, batch_size=1, shuffle=False)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay)

print("---------------------Start Training---------------------")
if opt.demo == False:
    log_start_train()
    start_epoch = opt.load_epoch + 1
    epochs = opt.nepoch
    for epoch in range(start_epoch, start_epoch+epochs):
        for idx, gt_data in enumerate(pix_loader):
            for item in gt_data:
                if item != 'sequence_id':
                    gt_data[item] = gt_data[item].to(opt.device)
            gt_data['mesh_points'] = gt_data[ 'mesh_points'].float()

            mesh_coordinates_results, points_from_edges, point_indicators, output_edges, boundary_point_ids, faces  = net(gt_data['img'], gt_data['cls'], threshold=opt.threshold, factor=opt.factor)
            est_data = {'mesh_coordinates_results':mesh_coordinates_results, 'points_from_edges':points_from_edges,
                        'point_indicators':point_indicators, 'output_edges':output_edges, 'boundary_point_ids':boundary_point_ids.bool(), 'faces':faces}
            loss_dict = mgn_loss(est_data, gt_data)
            loss = loss_dict['total']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % opt.log_freq == 0:
                message = '( epoch: %d, ) ' % (epoch)
                message += '( step: %d, ) ' % (idx)
                message += ' %s: %.5f ' % ("loss_train_total", loss.item())
                message += ' %s: %.5f ' % ("chanfer_loss_total", loss_dict["chamfer_loss"])
                message += ' %s: %.5f ' % ('edge_loss', loss_dict['edge_loss'])
                message += ' %s: %.5f ' % ('face_loss', loss_dict['face_loss'])
                message += ' %s: %.5f ' % ('boundary_loss', loss_dict['boundary_loss'])
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
            if idx % opt.check_freq == 0:
                print('epoch {} step {} loss: {:.4f} removed edge: {}'.format(epoch, idx, loss.item(), torch.sum(boundary_point_ids).item()))

        if epoch % opt.save_freq == 0:
            print("saving net...")
            if not os.path.exists(os.path.join(opt.model_path, opt.name)):
                os.makedirs(os.path.join(opt.model_path, opt.name))
            model_path = opt.model_path + '/'+ opt.name + '/model_epoch{}.pth'.format(epoch)
            torch.save(net.state_dict(), model_path)
            print("network saved")

else:
    import open3d as o3d
    net.eval()
    num = 1
    with torch.no_grad():
         for idx, gt_data in enumerate(pix_loader):
            if num > opt.demo_num:
                break 
            gt_data['mesh_points'] = gt_data['mesh_points'].float()
            point,_,_,_,_,faces = net(gt_data['img'], gt_data['cls'], train=False, threshold=opt.threshold, factor=opt.factor)
            point = point[-1][0]
            point[2,:] *= -1
            point = point.permute(1,0)
            point = point.cpu().numpy()
            faces = faces[0]
            faces = faces - 1
            import pymesh
            pymesh.save_mesh_raw(opt.demo_path + '/example{}.ply'.format(idx), point, faces)

            y_point = gt_data['mesh_points'][0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(y_point)
            o3d.io.write_point_cloud(opt.demo_path + '/ground_truth{}.ply'.format(idx), pcd)
            # mesh = net.generate_mesh(gt_data['img'], gt_data['cls'])
            # colormap = mesh_processor.ColorMap()
            # mesh_processor.save(mesh, opt.demo_path + '/example.ply', colormap)
            print("mesh saved at " + opt.demo_path + '/example{}.ply'.format(idx))

            imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
            imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
            img = gt_data['img'][0] * imagenet_std + imagenet_mean
            import torchvision.transforms as transforms
            img = transforms.ToPILImage()(img)
            img.save(opt.demo_path + '/example{}.png'.format(idx))
            num += 1
    




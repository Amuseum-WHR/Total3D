import argparse
import os
import datetime
import time
from easydict import EasyDict
import model.MGN.MGN_model as model
from model.loss import SVRLoss
import dataset.Pix3dDataloader as pixed_loader
import torch
from torch.utils.data import DataLoader

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

    parser.add_argument("--nepoch", type = float, default = 400, help = 'the total training epochs')

    parser.add_argument("--model_path", type = str, default = "out", help = 'path of saved model')
    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "test_code", help = 'name of this training process')
    parser.add_argument("--save_freq", type = int, default = 100)
    parser.add_argument("--check_freq", type = int, default = 4)
    
    
    parser.add_argument("--demo", type = bool, default = False, help = 'demo or not')
    parser.add_argument("--demo_path", type = str, default = 'demo')

    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt

opt = parser()
if torch.cuda.is_available():
    opt.device = torch.device(f"cuda:1")
else:
    opt.device = torch.device(f"cpu")

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
dataset_train = pixed_loader.PixDataset(device=opt.device)

if opt.demo == False:
    pixed_loader = DataLoader(dataset_train, batch_size=2, shuffle=True)
else:
    pixed_loader = DataLoader(dataset_train, batch_size=1, shuffle=False)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay)

if opt.demo == False:
    log_start_train()
    epochs = opt.nepoch
    for epoch in range(epochs):
        for idx, gt_data in enumerate(pixed_loader):
            for item in gt_data:
                gt_data[item].to(opt.device)
            mesh_coordinates_results, points_from_edges, point_indicators, output_edges, boundary_point_ids, faces  = net(gt_data['img'], gt_data['cls'])
            est_data = {'mesh_coordinates_results':mesh_coordinates_results, 'points_from_edges':points_from_edges,
                        'point_indicators':point_indicators, 'output_edges':output_edges, 'boundary_point_ids':boundary_point_ids, 'faces':faces}
            loss_dict = mgn_loss(est_data, gt_data)
            loss = loss_dict['total']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            message = '( epoch: %d, ) ' % (epoch)
            message += '( step: %d, ) ' % (idx)
            message += '%s: %.5f' % ("loss_train_total", loss.item())
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)
        if epoch % opt.check_freq == 0:
            print('epoch {} loss: {:.4f}'.format(epoch, loss.item()))

        if epoch % opt.save_freq == 0:
            print("saving net...")
            if not os.path.exists(os.path.join(opt.model_path, opt.name)):
                os.makedirs(os.path.join(opt.model_path, opt.name))
            model_path = opt.model_path + '/'+ opt.name + '/model_epoch{}.pth'.format(epoch)
            torch.save(net.state_dict(), model_path)
            print("network saved")

else:
    import open3d as o3d
    net.load_state_dict(torch.load(opt.model_path + '/'+ opt.name + '/model_epoch399.pth'))
    net.eval()
    with torch.no_grad():
         for idx, gt_data in enumerate(pixed_loader):
            _,point,_,_,_,_ = net(gt_data['img'], gt_data['cls'])
            point = point[-1][0]
            point = point.permute(1,0)
            point = point.cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point)
            o3d.io.write_point_cloud(opt.demo_path + '/example{}.ply'.format(idx), pcd)
            # mesh = net.generate_mesh(gt_data['img'], gt_data['cls'])
            # colormap = mesh_processor.ColorMap()
            # mesh_processor.save(mesh, opt.demo_path + '/example.ply', colormap)
            print("mesh saved at " + opt.demo_path + '/example{}.ply'.format(idx))
    




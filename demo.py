from tester import Tester

import time
import datetime
import argparse
import os

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from dataset.SunrgbdDataloader import SunDataset, collate_fn
from configs.data_config import Config as Data_Config

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
    parser.add_argument("--threshold", type = float, default = 0.2, help='threshold of tnn network')
    parser.add_argument("--factor", type = float, default = 0.5, help='factor of tnn network')

    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--betas", type = list, default = [0.9, 0.999])
    parser.add_argument("--eps", type = float, default = 1e-08)
    parser.add_argument("--weight_decay", type = float, default = 1e-04)
    parser.add_argument("--batch_size", type = int, default = 1, help = 'Batch Size' )
    parser.add_argument("--nepoch", type = float, default = 1, help = 'the total training epochs')

    parser.add_argument("--mgn_load_path", type = str, default = "out/mgn_pretrain_model.pth", help = 'path of saved mgn model')
    parser.add_argument("--len_load_path", type = str, default = "out/len_pretrain_model.pth", help = 'path of saved odn model')
    parser.add_argument("--odn_load_path", type = str, default = "out/odn_pretrain_model.pth", help = 'path of saved len model')
    parser.add_argument("--t3d_load_path", type = str, default = "", help = 'path of saved t3d model')
    parser.add_argument("--model_path", type=str, default="out", help='dir to save checkpoints')

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
    dataset = SunDataset(root_path='.', device= opt.device, mode='test')
    Train_loader = DataLoader(dataset, batch_size= opt.batch_size, collate_fn=collate_fn, shuffle = True)
    trainer = Tester(opt, device=opt.device)
    trainer.model.to(opt.device)

    trainer.model.eval()
    loop = tqdm(enumerate(Train_loader), total=len(Train_loader))
    for idx, gt_data in loop:
        with torch.no_grad():
            est_data, data = trainer.train_step(gt_data,train=False)
            break

    from model.utils.libs import get_layout_bdb_sunrgbd, get_rotation_matix_result, get_bdb_evaluation, write_obj, to_dict_tensor
    from scipy.io import savemat
    
    dataset_config = Data_Config('sunrgbd')
    bins_tensor = to_dict_tensor(dataset_config.bins, if_cuda=True)

    lo_bdb3D_out = get_layout_bdb_sunrgbd(bins_tensor, est_data['lo_ori_reg'],
                                          torch.argmax(est_data['lo_ori_cls'], 1),
                                          est_data['lo_centroid'],
                                          est_data['lo_coeffs'])
    
    cam_R_out = get_rotation_matix_result(bins_tensor,
                                          torch.argmax(est_data['pitch_cls'], 1), est_data['pitch_reg'],
                                          torch.argmax(est_data['roll_cls'], 1), est_data['roll_reg'])
    
    P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 -
                            (data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * est_data['offset_2D'][:, 0],
                            (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 -
                            (data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * est_data['offset_2D'][:,1]), 1)
    
    bdb3D_out_form_cpu, bdb3D_out = get_bdb_evaluation(bins_tensor,
                                                       torch.argmax(est_data['ori_cls'], 1),
                                                       est_data['ori_reg'],
                                                       torch.argmax(est_data['centroid_cls'], 1),
                                                       est_data['centroid_reg'],
                                                       data['size_cls'], est_data['size_reg'], P_result,
                                                       data['K'], cam_R_out, data['split'], return_bdb=True)
    
        # save results
    print("Saving...")
    nyu40class_ids = [int(evaluate_bdb['classid']) for evaluate_bdb in bdb3D_out_form_cpu]
    save_path = opt.demo_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save layout
    savemat(os.path.join(save_path, 'layout.mat'),
            mdict={'layout': lo_bdb3D_out[0, :, :].cpu().numpy()})
    # save bounding boxes and camera poses
    interval = data['split'][0].cpu().tolist()
    current_cls = nyu40class_ids[interval[0]:interval[1]]

    savemat(os.path.join(save_path, 'bdb_3d.mat'),
            mdict={'bdb': bdb3D_out_form_cpu[interval[0]:interval[1]], 'class_id': current_cls})
    savemat(os.path.join(save_path, 'r_ex.mat'),
            mdict={'cam_R': cam_R_out[0, :, :].cpu().numpy()})
    # # save meshes
    current_faces = est_data['out_faces'][interval[0]:interval[1]].cpu().numpy()
    current_coordinates = est_data['meshes'].transpose(1, 2)[interval[0]:interval[1]].cpu().numpy()
    

    # import pymesh
    for obj_id, obj_cls in enumerate(current_cls):
        # if obj_id >= len(current_coordinates):
        #     print('more....')
        file_path = os.path.join(save_path, '%s_%s.obj' % (obj_id, obj_cls))

            # print("saving " + NYU40CLASSES[obj_cls])
        mesh_obj = {'v': current_coordinates[obj_id],
                    'f': current_faces[obj_id]}
        write_obj(file_path, mesh_obj)
        # pymesh.save_mesh_raw(file_path, current_coordinates[obj_id], current_faces[obj_id])

    img_path = os.path.join(save_path, 'exp_img.png')
    img = data['image'][0].to("cpu")
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img = img * imagenet_std + imagenet_mean
    import torchvision.transforms as transforms
    img = transforms.ToPILImage()(img)
    img.save(img_path)

    import numpy as np

    total_vertice = None
    total_face = None

    # for obj_id, obj_cls in enumerate(current_cls):
    #     points = current_coordinates[obj_id]
    #     mesh_center = (points.max(0) + points.min(0)) / 2.
    #     points = points - mesh_center

    #     mesh_coef = (points.max(0) - points.min(0)) / 2.
    #     points = points.dot(np.diag(1./mesh_coef)).dot(np.diag(bdb3D_out_form_cpu[obj_id]['coeffs']))

    #     # set orientation
    #     points = points.dot(bdb3D_out_form_cpu[obj_id]['basis'])

    #     # move to center
    #     points = points + bdb3D_out_form_cpu[obj_id]['centroid']

    #     # file_path = os.path.join(save_path, '%s_%s_666.obj' % (obj_id, obj_cls))
    #     # pymesh.save_mesh_raw(file_path, points, current_faces[obj_id])
    #     # print(points.shape)
    #     if obj_id == 0:
    #         total_vertice = points
    #         total_face = current_faces[obj_id]
    #     else:
    #         total_vertice = np.vstack((total_vertice, points))
    #         faces = current_faces[obj_id]
    #         faces = faces + obj_id * 2562
    #         total_face = np.vstack((total_face, faces))
    
    # file_path = '%s/recon.ply' % (save_path)
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(total_vertice)
    # o3d.io.write_point_cloud(file_path, pcd)
    # import pymesh
    # pymesh.save_mesh_raw(file_path, total_vertice, total_face)

    print("Visualizeing......")
    import scipy.io as sio
    from utils.visualize import format_bbox, format_layout, format_mesh, Box
    from glob import glob
    import numpy as np

    pre_layout_data = sio.loadmat(os.path.join(save_path, 'layout.mat'))['layout']
    pre_box_data = sio.loadmat(os.path.join(save_path, 'bdb_3d.mat'))

    pre_boxes = format_bbox(pre_box_data, 'prediction')
    pre_layout = format_layout(pre_layout_data)
    pre_cam_R = sio.loadmat(os.path.join(save_path, 'r_ex.mat'))['cam_R']

    vtk_objects, pre_boxes = format_mesh(glob(os.path.join(save_path, '*.obj')), pre_boxes)

    image = np.array(img)
    cam_K = data['K'][0].to("cpu")

    scene_box = Box(image, None, cam_K, None, pre_cam_R, None, pre_layout, None, pre_boxes, 'prediction', output_mesh = vtk_objects)
    scene_box.draw_projected_bdb3d('prediction', if_save=True, save_path = '%s/3dbbox.png' % (save_path))
    scene_box.draw3D(if_save=True, save_path = '%s/recon.png' % (save_path))
        

    

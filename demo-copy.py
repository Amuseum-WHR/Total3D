from tester import Tester

import time
import datetime
import argparse
import os
import json

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from dataset.SunrgbdDataloader import SunDataset, collate_fn
from configs.data_config import NYU40CLASSES
# from detection import TwoDBB
from PIL import Image
import numpy as np

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

    parser.add_argument("--mgn_load_path", type = str, default = "", help = 'path of saved mgn model')
    parser.add_argument("--len_load_path", type = str, default = "") # "out/len_pretrain_model.pth", help = 'path of saved odn model')
    parser.add_argument("--odn_load_path", type = str, default = "") # "out/odn_pretrain_model.pth", help = 'path of saved len model')
    parser.add_argument("--t3d_load_path", type = str, default = "out/t3d_checkpoints_epoch66.pth") # "out/t3d_checkpoints_epoch66.pth"), help = 'path of saved t3d model')
    parser.add_argument("--model_path", type=str, default="out", help='dir to save checkpoints')

    parser.add_argument("--log_path", type = str, default = "log", help = 'path of log info')
    parser.add_argument("--name", type = str, default = "total3d", help = 'name of this training process')
    
    parser.add_argument("--demo", action="store_true", default = False, help = 'demo or not')
    parser.add_argument("--demo_path", type = str, default = 'demo')
    parser.add_argument("--check_freq", type = int, default = 5, help = 'The frequency of print loss in screen.')
    parser.add_argument("--save_freq", type = int, default = 10, help = 'The frequency of saving a model.')
    parser.add_argument("--cuda", type =str, default = "cuda:0", help = 'Which GPU to use for training.')
    parser.add_argument("--cuda_num", type =int, default = 0, help = 'Which GPU to use for training.')

    parser.add_argument("--mode", type =str, default = 'normal', choices = ['normal', 'replace', 'add'], help = 'mode to run the code')
    parser.add_argument("--src_class", type =str, default = 'table', help = 'the class we want to replace')
    parser.add_argument("--target_class", type =str, default = 'sofa', help = 'the class we want to replace with')
    parser.add_argument('--detection_path', type =str, default='./detection-pretrain/sunrgbd_model_95000.npz')
    parser.add_argument('--img_path', type =str, default='demo_img/720.jpg')
    parser.add_argument('--add_img', type =str, default='single_object/test460.jpg')
    parser.add_argument('--add_box', type =list, default=[38, 304, 245, 496])
    parser.add_argument('--k', type =str, default='default', choices=['default','ours','txt'])
    parser.add_argument('--k_path', type =str, default='K.txt')
    
    opt = parser.parse_args()
    opt = EasyDict(opt.__dict__)

    opt.date = str(datetime.datetime.now())
    return opt

def get_random_data_from_sunrgbd(save_path = None):
    dataset = SunDataset(root_path='.', device= opt.device, mode='test')
    Train_loader = DataLoader(dataset, batch_size= opt.batch_size, collate_fn=collate_fn, shuffle = True)
    for idx, gt_data in enumerate(Train_loader):
        if idx == 0:
            break

    if save_path is not None:
        with open(os.path.join(save_path, 'detections.json'), 'w') as f:
            n_objects = gt_data['boxes_batch']['bdb2D_pos'].shape[0]
            size_cls = gt_data['boxes_batch']['size_cls'].tolist()
            labels = [int(np.argmax(one_hot)) for one_hot in size_cls]
            detections = []
            for i in range(n_objects):
                detections.append({'bbox': gt_data['boxes_batch']['bdb2D_pos'][i].tolist(), 'class': NYU40CLASSES[labels[i]]})
            f.write(json.dumps(detections))
    return gt_data


if __name__ == "__main__":
    opt = parser()
    if torch.cuda.is_available():
        opt.device = torch.device(opt.cuda)
        torch.cuda.set_device(opt.cuda_num) 
    else:
        opt.device = torch.device("cpu")
    
    tester = Tester(opt, device=opt.device)
    tester.model.to(opt.device)
    tester.model.eval()

    if opt.k == 'default':
        K = torch.FloatTensor([[529.5,   0.,  365.],
                                [0.,   529.5,  265.], 
                                [0.,     0.,     1. ]]
                                )
    elif opt.k == 'ours':
        # K = torch.FloatTensor([[2961, 0, 1079], 
        #                         [0, 2962, 1933], 
        #                         [0, 0, 1]])
        # K = torch.FloatTensor([[3453.80723446789, 0, 1509.77575786894], 
        #                         [0, 3450.11438003616, 2042.44171449600], 
        #                         [0, 0, 1]])
        # K = torch.FloatTensor([[3346.90380668512, 0, 2035.77624651872], 
        #                         [0, 3352.41685350737, 1553.89162781762], 
        #                         [0, 0, 1]])
        K = torch.FloatTensor([[ 1.42278701e+03, -4.60370724e-01,  6.62377447e+02],
                                [ 0.00000000e+00,  1.42106838e+03,  3.03415123e+02],
                                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    else:
        txt_path = opt.k_path
        K = torch.FloatTensor(np.loadtxt(txt_path))

    # gt_data = tester.read_from_img(K)
    
    now_time = str(datetime.datetime.now().replace(microsecond=0)).replace(' ','_').replace(':','-')
    save_path = os.path.join(opt.demo_path, now_time)
    os.mkdir(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gt_data = get_random_data_from_sunrgbd(save_path)
    # gt_data = tester.get_data_from_json('demo/inputs/2/img.jpg', 'demo/inputs/2/detections.json', K)
    with torch.no_grad():
        est_data, data = tester.step(gt_data)

    lo_bdb3D_out, cam_R_out, bdb3D_out_form_cpu, bdb3D_out = tester.calculate(est_data, data)

    # print(lo_bdb3D_out_form_cpu)
    # print(bdb3D_out)
    # print(bdb3D_out_form_cpu)
    # print(lo_bdb3D_out-data['lo_bdb3D'])
    # print(bdb3D_out[0]-data['bdb3D'][0])
    # bdb3D_out_form_cpu.append(lo_bdb3D_out_form_cpu)
    
    # save results
    print("Saving...")
    nyu40class_ids = [int(evaluate_bdb['classid']) for evaluate_bdb in bdb3D_out_form_cpu]

    from model.utils.libs import write_obj
    from scipy.io import savemat
    
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

    print(current_faces)
    for obj_id, obj_cls in enumerate(current_cls):
        file_path = os.path.join(save_path, '%s_%s.obj' % (obj_id, obj_cls))
        mesh_obj = {'v': current_coordinates[obj_id],
                    'f': current_faces[obj_id]}
        write_obj(file_path, mesh_obj)

    img_path = os.path.join(save_path, 'img.jpg')
    img = gt_data['origin_image'][0].to("cpu")
    import torchvision.transforms as transforms
    img = transforms.ToPILImage()(img)
    img.save(img_path)

    file_path = '%s/recon.ply' % (save_path)
    # tester.save_mesh(current_coordinates, current_faces, bdb3D_out_form_cpu[:-1], current_cls[:-1], file_path)

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
    # print(len(pre_boxes))

    img = np.array(img)
    cam_K = data['K'][0].to("cpu")
    image = img
    # image = np.uint8(np.zeros((img.shape[0]*3, img.shape[1]*3, img.shape[2])))

    scene_box = Box(image, None, cam_K, None, pre_cam_R, None, pre_layout, None, pre_boxes, 'prediction', output_mesh = vtk_objects)
    scene_box.draw_projected_bdb3d('prediction', if_save=True, save_path = '%s/3dbbox.png' % (save_path))
    scene_box.draw3D(if_save=True, save_path = '%s/recon.png' % (save_path))
        

    

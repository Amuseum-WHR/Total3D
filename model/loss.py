'''
Loss Functions
Significantly Based on Total3DUnderstanding's models/loss.py
'''

import torch
import torch.nn as nn
import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D

binary_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')
reg_criterion = nn.SmoothL1Loss(reduction='mean')
cls_criterion = nn.CrossEntropyLoss(reduction='mean')


def cls_reg_loss(est_cls, gt_cls, est_reg, gt_reg):
    cls_loss = cls_criterion(est_cls, gt_cls)
    if len(est_reg.size()) == 3:
        est_reg = torch.gather(est_reg, 1, gt_cls.view(gt_reg.size(0), 1, 1).expand(gt_reg.size(0), 1, gt_reg.size(1)))
    else:
        est_reg = torch.gather(est_reg, 1, gt_cls.view(gt_reg.size(0), 1).expand(gt_reg.size(0), 1))
    est_reg = est_reg.squeeze(1)  # Only Calculate the dimension of feature where cls is 1 in goundtruth.     
    reg_loss = reg_criterion(est_reg, gt_reg) 
    return cls_loss, reg_loss

class BaseLoss(object):
    '''base loss class'''
    def __init__(self, weight=1):
        '''initialize loss module'''
        self.weight = weight


class SVRLoss(BaseLoss):
    def __call__(self, est_data, gt_data, subnetworks, face_sampling_rate):
        '''
        est_data: prediction result
        gt_data: truth ground
        subnetworks: num of subnetworks of tmn_networks
        face_sampling_rate: face_samples

        dict of gt_data:
        {'sequence_id':sequence['sample_id'],
            'img':data_transforms(image),
            'cls':cls_codes,
            'mesh_points':gt_points,
            'densities': densities}
        '''
        device = est_data['mesh_coordinates_results'][0].device
        # chamfer losses
        chamfer_loss = torch.tensor(0.).to(device)
        edge_loss = torch.tensor(0.).to(device)
        boundary_loss = torch.tensor(0.).to(device)

        for stage_id, mesh_coordinates_result in enumerate(est_data['mesh_coordinates_results']):
            mesh_coordinates_result = mesh_coordinates_result.transpose(1, 2)
            # points to points chamfer loss
            dist_chamfer = dist_chamfer_3D.chamfer_3DDist()
            dist1, dist2 = dist_chamfer(gt_data['mesh_points'], mesh_coordinates_result)[:2]
            chamfer_loss += (torch.mean(dist1)) + (torch.mean(dist2))

            # boundary loss
            if stage_id == subnetworks - 1:
                if 1 in est_data['boundary_point_ids']:
                    boundary_loss = torch.mean(dist2[est_data['boundary_point_ids']])

            # edge loss
            edge_vec = torch.gather(mesh_coordinates_result, 1,
                                    (est_data['output_edges'][:, :, 0] - 1).unsqueeze(-1).expand(est_data['output_edges'].size(0),
                                                                                     est_data['output_edges'].size(1), 3)) \
                       - torch.gather(mesh_coordinates_result, 1,
                                      (est_data['output_edges'][:, :, 1] - 1).unsqueeze(-1).expand(est_data['output_edges'].size(0),
                                                                                       est_data['output_edges'].size(1), 3))

            edge_vec = edge_vec.view(edge_vec.size(0) * edge_vec.size(1), edge_vec.size(2))
            edge_loss += torch.mean(torch.pow(torch.norm(edge_vec, p=2, dim=1), 2))

        chamfer_loss = 100 * chamfer_loss / len(est_data['mesh_coordinates_results'])
        edge_loss = 100 * edge_loss / len(est_data['mesh_coordinates_results'])
        boundary_loss = 100 * boundary_loss

        # face distance losses
        face_loss = torch.tensor(0.).to(device)
        for points_from_edges_by_step, points_indicator_by_step in zip(est_data['points_from_edges'], est_data['point_indicators']):
            points_from_edges_by_step = points_from_edges_by_step.transpose(1, 2).contiguous()
            _, dist2_face, _, idx2 = dist_chamfer(gt_data['mesh_points'], points_from_edges_by_step)
            idx2 = idx2.long()
            dist2_face = dist2_face.view(dist2_face.shape[0], dist2_face.shape[1] // face_sampling_rate,
                                         face_sampling_rate)

            # average distance to nearest face.
            dist2_face = torch.mean(dist2_face, dim=2)
            local_dens = gt_data['densities'][:, idx2[:]][range(gt_data['densities'].size(0)), range(gt_data['densities'].size(0)), :]
            in_mesh = (dist2_face <= local_dens).float()
            face_loss += binary_cls_criterion(points_indicator_by_step, in_mesh)

        if est_data['points_from_edges']:
            face_loss = face_loss / len(est_data['points_from_edges'])

        return {'chamfer_loss': chamfer_loss, 'face_loss': 0.01 * face_loss,
                'edge_loss': 0.1 * edge_loss, 'boundary_loss': 0.5 * boundary_loss}

class Detection_Loss(BaseLoss):
    def __init__(self, weight = 1, cls_reg_ratio = 10):
        super(BaseLoss, self).__init__(weight = weight)
        self.cls_reg_ratio = cls_reg_ratio

    def __call__(self, est_data, gt_data):
        size_reg_loss = reg_criterion(est_data['size_reg'], gt_data['size_reg']) * self.cls_reg_ratio
        ori_cls_loss, ori_reg_loss = cls_reg_loss(est_data['ori_cls'], gt_data['ori_cls'], est_data['ori_reg'], gt_data['ori_cls'])
        centroid_cls_loss, centroid_reg_loss = cls_reg_loss(est_data['centroid_cls'], gt_data['centroid_cls'], est_data['centroid_reg'], gt_data['centroid_reg'])
        offset_reg_loss = reg_criterion(est_data['offset_2D'], gt_data['offset_2D'])

        total_loss = size_reg_loss + ori_cls_loss + ori_reg_loss + centroid_cls_loss + centroid_reg_loss + offset_reg_loss
        return {'total': total_loss, 
                'size_reg_loss': size_reg_loss, 'ori_cls_loss': ori_cls_loss, 'ori_reg_loss': ori_reg_loss,
                'centroid_cls_loss': centroid_cls_loss, 'centroid_reg_loss': centroid_reg_loss,
                'offset_reg_loss': offset_reg_loss}
        
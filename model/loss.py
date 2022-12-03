'''
Loss Functions
Significantly Based on Total3DUnderstanding's models/loss.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points

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

def _handle_pointcloud_input(
    points,
    lengths,
    normals
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals

def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, norm=2):

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    x_idx = x_nn.idx[..., 0]
    y_idx = y_nn.idx[..., 0]

    return cham_x, cham_y, x_idx, y_idx


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
            dist1, dist2 = chamfer_distance(gt_data['mesh_points'], mesh_coordinates_result)[:2]
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
            _, dist2_face, _, idx2 = chamfer_distance(gt_data['mesh_points'], points_from_edges_by_step)
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

        print('new: ', 'chamfer_loss', chamfer_loss, 'face_loss', 0.01 * face_loss, 'edge_loss', 0.1 * edge_loss, 'boundary_loss', 0.5 * boundary_loss)

        return {'chamfer_loss': chamfer_loss, 'face_loss': 0.01 * face_loss,
                'edge_loss': 0.1 * edge_loss, 'boundary_loss': 0.5 * boundary_loss}

        chamfer_loss = torch.tensor(0.).to(device)
        edge_loss = torch.tensor(0.).to(device)
        boundary_loss = torch.tensor(0.).to(device)
        import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D

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
        
        print('old: ', 'chamfer_loss', chamfer_loss, 'face_loss', 0.01 * face_loss, 'edge_loss', 0.1 * edge_loss, 'boundary_loss', 0.5 * boundary_loss)

        return {'chamfer_loss': chamfer_loss, 'face_loss': 0.01 * face_loss,
                'edge_loss': 0.1 * edge_loss, 'boundary_loss': 0.5 * boundary_loss}

class Detection_Loss(BaseLoss):
    def __init__(self, weight = 1, cls_reg_ratio = 10):
        super(Detection_Loss, self).__init__(weight = weight)
        self.cls_reg_ratio = cls_reg_ratio

    def __call__(self, est_data, gt_data):
        size_reg_loss = reg_criterion(est_data['size_reg'], gt_data['size_reg']) * self.cls_reg_ratio
        ori_cls_loss, ori_reg_loss = cls_reg_loss(est_data['ori_cls'], gt_data['ori_cls'], est_data['ori_reg'], gt_data['ori_cls'])
        centroid_cls_loss, centroid_reg_loss = cls_reg_loss(est_data['centroid_cls'], gt_data['centroid_cls'], est_data['centroid_reg'], gt_data['centroid_reg'])
        offset_reg_loss = reg_criterion(est_data['offset_2D'], gt_data['offset_2D'])

        total_loss = size_reg_loss + ori_cls_loss + ori_reg_loss + centroid_cls_loss + centroid_reg_loss + offset_reg_loss
        print("total:", total_loss)
        return {'total': total_loss, 
                'size_reg_loss': size_reg_loss, 'ori_cls_loss': ori_cls_loss, 'ori_reg_loss': ori_reg_loss,
                'centroid_cls_loss': centroid_cls_loss, 'centroid_reg_loss': centroid_reg_loss,
                'offset_reg_loss': offset_reg_loss}
        
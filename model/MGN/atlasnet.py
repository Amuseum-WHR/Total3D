import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from model.MGN.miscs import sample_points_on_edges


class EREstimate(nn.Module):
    def __init__(self, bottleneck_size=2500, output_dim = 3):
        super(EREstimate, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500, output_dim = 3):
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

def generate_square(grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        grain = int(grain)
        grain = grain - 1  # to return grain*grain points
        # generate regular grid
        faces = []
        vertices = []
        for i in range(0, int(grain + 1)):
            for j in range(0, int(grain + 1)):
                vertices.append([i / grain, j / grain, 0])

        for i in range(1, int(grain + 1)):
            for j in range(0, (int(grain + 1) - 1)):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i + 1,
                              j + (grain + 1) * (i - 1)])
        for i in range(0, (int((grain + 1)) - 1)):
            for j in range(1, int((grain + 1))):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i - 1,
                              j + (grain + 1) * (i + 1)])

        return np.array(vertices), np.array(faces)

def load_template(number):
    file_name = './model/MGN/sphere%d.pkl' % (number)

    with open(file_name, 'rb') as file:
        sphere_obj = pickle.load(file)
        sphere_points_normals = torch.from_numpy(sphere_obj['v']).float()
        sphere_faces = torch.from_numpy(sphere_obj['f']).long()
        sphere_adjacency = torch.from_numpy(sphere_obj['adjacency'].todense()).long()
        sphere_edges = torch.from_numpy(sphere_obj['edges']).long()
        sphere_edge2face = torch.from_numpy(sphere_obj['edge2face'].todense()).type(torch.uint8)
    return sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face

class Atlasnet(nn.Module):

    def __init__(self, opt):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt: 
        """
        super(Atlasnet, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.subnetworks = self.opt.subnetworks
        self.face_samples = self.opt.face_samples

        # new try

        self.decoders = nn.ModuleList(
            [PointGenCon(3 + self.opt.bottleneck_size + self.opt.num_classes) for i in range(0, self.subnetworks)])
        # for i in range(0, self.subnetworks):
        #     self.decoders[i] = self.decoders[i].to(self.device)
        self.error_estimators = nn.ModuleList(
                [EREstimate(bottleneck_size = 3 + self.opt.bottleneck_size + self.opt.num_classes, output_dim=1) for i in range(0, max(self.subnetworks-1, 1))])
        # for i in range(0, max(self.subnetworks-1, 1)):
        #     self.error_estimators[i] = self.error_estimators[i].to(self.device)


    def init_error_estimator(self):
        self.error_estimators = nn.ModuleList(
                [EREstimate(bottleneck_size = 3 + self.opt.bottleneck_size + self.opt.num_classes, output_dim=1) for i in range(0, max(self.subnetworks-1, 1))])
        return
    
    def freeze_tmn(self):
        for param in self.error_estimators[-1].parameters():
                param.requires_grad = False
        return


    def forward(self, latent_vector, train=True, threshold = 0.25, factor = 1):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        threshold = self.opt.threshold
        factor = self.opt.factor
        device = self.device

        sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face = load_template(self.opt.number_points)

        if train:
            mode = 'train'
            current_faces = None
        else:
            mode = 'test'
            current_faces = sphere_faces.clone().unsqueeze(0).to(device)
            current_faces = current_faces.repeat(latent_vector.size(0), 1, 1)

        current_edges = sphere_edges.clone().unsqueeze(0).to(device)
        current_edges = current_edges.repeat(latent_vector.size(0), 1, 1)

        current_shape_grid = sphere_points_normals[:, :3].t().expand(latent_vector.size(0), 3, self.opt.number_points).to(device)
        
        boundary_point_ids = torch.zeros(size=(latent_vector.size(0), self.opt.number_points), dtype=torch.uint8).to(device)
        n_edges = sphere_edges.shape[0]
        remove_edges_list = []
        out_shape_points = []
        out_sampled_mesh_points = []
        out_indicators = []

        for i in range(0, self.subnetworks):
            current_image_grid = latent_vector.unsqueeze(2).expand(latent_vector.size(0), latent_vector.size(1),
                                                           current_shape_grid.size(2)).contiguous()
            current_image_grid = torch.cat((current_shape_grid, current_image_grid), 1).contiguous()
            new_shape_grid = self.decoders[i](current_image_grid)
            current_shape_grid = current_shape_grid + new_shape_grid

            out_shape_points.append(current_shape_grid)

            if i == self.subnetworks - 1 and self.subnetworks > 1:
                remove_edges_list = [item for item in remove_edges_list if len(item)]
                if remove_edges_list:
                    remove_edges_list = torch.unique(torch.cat(remove_edges_list), dim=0)
                    for batch_id in range(latent_vector.size(0)):
                        rm_edges = remove_edges_list[remove_edges_list[:, 0] == batch_id, 1]
                        if len(rm_edges) > 0:
                            rm_candidates, counts = torch.unique(sphere_edges[rm_edges], return_counts=True)
                            boundary_ids = counts < sphere_adjacency[rm_candidates - 1].sum(1)
                            boundary_point_ids[batch_id][rm_candidates[boundary_ids] - 1] = 1

                return out_shape_points, out_sampled_mesh_points, out_indicators, current_edges, boundary_point_ids, current_faces

            sampled_points = sample_points_on_edges(current_shape_grid, current_edges, quantity=self.face_samples, mode=mode)

            out_sampled_mesh_points.append(sampled_points)

            # preprare for face error estimation
            current_image_grid = latent_vector.unsqueeze(2).expand(latent_vector.size(0), latent_vector.size(1), sampled_points.size(2)).contiguous()
            current_image_grid = torch.cat((sampled_points, current_image_grid), 1).contiguous()

            # estimate the distance from deformed points to gt mesh.
            indicators = self.error_estimators[i](current_image_grid)
            indicators = indicators.view(latent_vector.size(0), 1, n_edges, self.face_samples)
            indicators = indicators.squeeze(1)
            indicators = torch.mean(indicators, dim=2)

            out_indicators.append(indicators)

            # remove faces and modify the topology
            remove_edges = torch.nonzero(torch.sigmoid(indicators) < threshold)
            remove_edges_list.append(remove_edges)

            for batch_id in range(latent_vector.size(0)):
                rm_edges = remove_edges[remove_edges[:, 0] == batch_id, 1]
                if len(rm_edges)>0:
                    # cutting edges in training
                    current_edges[batch_id][rm_edges, :] = 1
                    if mode == 'test':
                        current_faces[batch_id][sphere_edge2face[rm_edges].sum(0).type(torch.bool), :] = 1
            

            threshold *= factor

        return out_shape_points, out_sampled_mesh_points, out_indicators, current_edges, boundary_point_ids, current_faces

    '''def generate_mesh(self, latent_vector, train = False, threshold = 0.1, factor = 1):
        assert latent_vector.size(0)==1, "input should have batch size 1!"
        import pymesh

        device = self.device

        sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face = load_template(self.opt.number_points)

        current_edges = sphere_edges.clone().unsqueeze(0).to(device)
        current_edges = current_edges.repeat(latent_vector.size(0), 1, 1)
        if train:
            mode = 'train'
            current_faces = None
        else:
            mode = 'test'
            current_faces = sphere_faces.clone().unsqueeze(0).to(device)
            current_faces = current_faces.repeat(latent_vector.size(0), 1, 1)

        current_shape_grid = sphere_points_normals[:, :3].t().expand(latent_vector.size(0), 3, self.opt.number_points).to(device)
        
        boundary_point_ids = torch.zeros(size=(latent_vector.size(0), self.opt.number_points), dtype=torch.uint8).to(device)
        n_edges = sphere_edges.shape[0]
        remove_edges_list = []
        out_shape_points = []
        out_sampled_mesh_points = []
        out_indicators = []

        for i in range(0, self.subnetworks):
            current_image_grid = latent_vector.unsqueeze(2).expand(latent_vector.size(0), latent_vector.size(1),
                                                           current_shape_grid.size(2)).contiguous()
            current_image_grid = torch.cat((current_shape_grid, current_image_grid), 1).contiguous()
            new_shape_grid = self.decoders[i](current_image_grid)
            current_shape_grid = current_shape_grid + new_shape_grid

            out_shape_points.append(current_shape_grid)

            if i == self.subnetworks - 1 and self.subnetworks > 1:
                remove_edges_list = [item for item in remove_edges_list if len(item)]
                if remove_edges_list:
                    remove_edges_list = torch.unique(torch.cat(remove_edges_list), dim=0)
                    for batch_id in range(latent_vector.size(0)):
                        rm_edges = remove_edges_list[remove_edges_list[:, 0] == batch_id, 1]
                        if len(rm_edges) > 0:
                            rm_candidates, counts = torch.unique(sphere_edges[rm_edges], return_counts=True)
                            boundary_ids = counts < sphere_adjacency[rm_candidates - 1].sum(1)
                            boundary_point_ids[batch_id][rm_candidates[boundary_ids] - 1] = 1

                break

            sampled_points = sample_points_on_edges(current_shape_grid, current_edges, quantity=self.face_samples, mode=mode)

            out_sampled_mesh_points.append(sampled_points)

            # preprare for face error estimation
            current_image_grid = latent_vector.unsqueeze(2).expand(latent_vector.size(0), latent_vector.size(1), sampled_points.size(2)).contiguous()
            current_image_grid = torch.cat((sampled_points, current_image_grid), 1).contiguous()

            # estimate the distance from deformed points to gt mesh.
            indicators = self.error_estimators[i](current_image_grid)
            indicators = indicators.view(latent_vector.size(0), 1, n_edges, self.face_samples)
            indicators = indicators.squeeze(1)
            indicators = torch.mean(indicators, dim=2)

            out_indicators.append(indicators)

            # remove faces and modify the topology
            remove_edges = torch.nonzero(torch.sigmoid(indicators) < threshold)
            remove_edges_list.append(remove_edges)

            for batch_id in range(latent_vector.size(0)):
                rm_edges = remove_edges[remove_edges[:, 0] == batch_id, 1]
                if len(rm_edges)>0:
                    # cutting edges in training
                    current_edges[batch_id][rm_edges, :] = 1
                    if mode == 'test':
                        current_faces[batch_id][sphere_edge2face[rm_edges].sum(0).type(torch.bool), :] = 1
            

            threshold *= factor

        # Deform return the deformed pointcloud
        out_points = out_sampled_mesh_points[-1][0].transpose(1, 0).cpu().numpy()
        out_faces = current_faces[0].cpu().numpy()

        # lenth = out_points.shape[0]
        # _, faces = generate_square(np.sqrt(lenth/2)+2)
        # faces = faces[0:lenth]
        mesh = pymesh.form_mesh(vertices=out_points, faces = out_faces)
        return mesh
        # return faces, out_points'''

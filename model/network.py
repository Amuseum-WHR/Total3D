import torch
import torch.nn as nn
import numpy as np

from model.MGN.MGN_model import EncoderDecoder as mgn_model
from model.ODN.ODN_model import ODN as odn_model
from model.LEN.LEN_model import PoseNet as len_model

from configs import data_config

class TOTAL3D(nn.Module):
    def __init__(self, opt):
        super(TOTAL3D, self).__init__()
        self.len = None
        cfg = data_config.Config('sunrgbd')
        self.odn = odn_model(cfg) # .to(opt.device)
        self.mgn = mgn_model(opt)
        self.len = len_model(cfg) # .to(opt.device)
        self.mgn_threshold = opt.threshold
        self.mgn_factor = opt.factor 

    def forward(self, len_input, odn_input, joint_input, train=True):
        l_est_data = self.len(len_input['image'])
        o_est_data = self.odn(odn_input['patch'], odn_input['g_features'], odn_input['split'], odn_input['rel_pair_counts'], odn_input['size_cls'])

        output_mesh,_ ,_ ,_ ,_ ,faces  = self.mgn(joint_input["patch_for_mesh"], joint_input["cls_codes_for_mesh"], self.mgn_threshold, self.mgn_factor)
        if train == True:
            output_mesh = output_mesh[-1]
            output_mesh[:, 2, :] *= -1
            m_est_data = {'meshes':output_mesh}
        else:
            out_faces = faces-1
            output_mesh = output_mesh[-1]
            output_mesh[:, 2, :] *= -1
            m_est_data = {'meshes':output_mesh, 'out_faces': out_faces}

        return l_est_data, o_est_data, m_est_data
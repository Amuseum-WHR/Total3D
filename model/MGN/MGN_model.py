from model.MGN.atlasnet import Atlasnet
# import auxiliary.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
# from auxiliary.ChamferDistancePytorch.fscore import fscore
import torch
import torch.nn as nn
import model.utils.resnet as resnet


class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        self.encoder = resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)

        self.decoder = Atlasnet(opt)
        self.to(opt.device)
        self.eval()

    def forward(self, x, size_cls, train=True, threshold=0.1, factor=1):
        latent = self.encoder(x)
        latent = torch.cat([latent, size_cls], 1)
        return self.decoder(latent, train=train, threshold=threshold, factor=factor)
    
    def freeze_tmn(self):
        self.decoder.freeze_tmn()
        return

    '''def generate_mesh(self, x, size_cls):
        latent = self.encoder(x)
        latent = torch.cat([latent, size_cls], 1)
        return self.decoder.generate_mesh(latent)'''
    
    '''def build_loss(self):
        loss_model = self.chamfer_loss
        return loss_model
    
    def chamfer_loss(self, pointsReconstructed, points, train=True):
        """
        Training loss of Atlasnet. The Chamfer Distance. Compute the f-score in eval mode.
        :return:
        """
        inCham1 = points.reshape(points.size(0), -1, 3).contiguous()
        inCham2 = pointsReconstructed.reshape(points.size(0), -1, 3).contiguous()

        distChamfer = dist_chamfer_3D.chamfer_3DDist()

        dist1, dist2, idx1, idx2 = distChamfer(inCham1, inCham2)  # mean over points
        loss = torch.mean(dist1) + torch.mean(dist2)  # mean over points
        if not train:
            loss_fscore, _, _ = fscore(dist1, dist2)
            loss_fscore = loss_fscore.mean()
            return loss, loss_fscore
        return loss'''


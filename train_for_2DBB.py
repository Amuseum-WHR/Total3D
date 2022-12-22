'''Reference: chainercv/examples/faster_rcnn/train.py'''
from __future__ import division

import argparse
import numpy as np
import os.path as osp
import datetime
import matplotlib
matplotlib.use('Agg')

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from chainercv import transforms

from dataset.Sunrgbd_2DBB_Dataloader import SunDataset_2DBB


sunrgbd_bbox_label_names = ('void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')


class Transform(object):

    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


def main():
    parser = argparse.ArgumentParser(
        description='Training Faster R-CNN for 2D Bounding Boxes on Sun RGB-D')
    parser.add_argument('--root_path', '-path', type=str, default=".")
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--out', '-o', default='./out')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=25000)
    parser.add_argument('--iteration', '-i', type=int, default=100000)
    parser.add_argument('--lr_shift', '-ls', type=float, default=0.5)
    parser.add_argument('--save_interval', '-si', type=int, default=5000)
    parser.add_argument('--evaluation_interval', '-ei', type=int, default=5000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_data = SunDataset_2DBB(args.root_path, mode='train')
    test_data = SunDataset_2DBB(args.root_path, mode='test')

    faster_rcnn = FasterRCNNVGG16(n_fg_class=len(sunrgbd_bbox_label_names),
                                  pretrained_model='imagenet')
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))

    train_data = TransformDataset(train_data, Transform(faster_rcnn))

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    now_time = str(datetime.datetime.today()).replace(' ','_')
    save_dir = osp.join(args.out, now_time)

    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=save_dir)

    weight_save_interval = args.save_interval, 'iteration'
    evaluation_interval = args.evaluation_interval, 'iteration'

    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn, 
                                   '2dbb_model_{.updater.iteration}.npz'),
        trigger=weight_save_interval)
    trainer.extend(extensions.ExponentialShift('lr', args.lr_shift),
                   trigger=(args.step_size, 'iteration'))

    log_interval = 20, 'iteration'
    plot_interval = 100, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/map',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model.faster_rcnn, use_07_metric=False,
            label_names=sunrgbd_bbox_label_names),
        trigger=evaluation_interval)

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()

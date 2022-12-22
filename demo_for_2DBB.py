'''Reference: chainercv/examples/faster_rcnn/demo.py'''
import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox


sunrgbd_bbox_label_names = ('void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop')


def TwoDBB(gpu, pretrained_model, image):
    model = FasterRCNNVGG16(
        n_fg_class=len(sunrgbd_bbox_label_names),
        pretrained_model=pretrained_model)

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    img = utils.read_image(image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=sunrgbd_bbox_label_names)
    plt.show()
    
    n_objects = bbox.shape[0]
    bbox_xy = [[int(bbox[i][1]), int(bbox[i][0]), int(bbox[i][3]), int(bbox[i][2])]  for i in range(n_objects)]
    dic = []
    for i in range(n_objects):
        dic.append({'bbox': bbox_xy[i], 'class': label[i]})
    return dic


def main():
    parser = argparse.ArgumentParser(
        description='2D Bounding Boxes demo')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pretrained-model', 
                        default='./pretrained_model/2dbb_model_95000.npz')
    parser.add_argument('--image', default='./demo/inputs/img1.jpg')
    args = parser.parse_args()

    print(TwoDBB(args.gpu, args.pretrained_model, args.image))


if __name__ == '__main__':
    main()

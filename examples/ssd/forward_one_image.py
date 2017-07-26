import argparse
import matplotlib.pyplot as plot
import yaml

import chainer

from chainercv.links import SSD300
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument(
        'label_names', help='The path to the yaml file with label names')
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='voc0712')
    args = parser.parse_args()

    with open(args.label_names, 'r') as f:
        label_names = tuple(yaml.load(f))

    model = SSD300(
        n_fg_class=len(label_names),
        pretrained_model=args.pretrained_model)
    model.score_thresh = 0.3
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score,
        label_names=label_names)
    plot.show()


if __name__ == '__main__':
    main()

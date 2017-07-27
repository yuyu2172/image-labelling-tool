import argparse
import json
import matplotlib.pyplot as plot
import os
import yaml

import chainer

from chainercv.links import SSD300
from chainercv import utils
from chainercv.visualizations import vis_bbox


def bbox_to_json(bbox, label, label_names, img_filename):
    """
    Note: The bbox follows OpenCV convention.

    bb (x_min, y_min, x_max, y_max)
    """

    d = {
        'complete': None,
        'image_filename': os.path.split(img_filename)[-1],
        'labels': []
    }
    for i, (bb, l) in enumerate(zip(bbox, label)):
        w = float(bb[3] - bb[1])
        h = float(bb[2] - bb[0])
        center_x = float(bb[1] + w / 2)
        center_y = float(bb[0] + h / 2)
        json_filename = os.path.splitext(img_filename)[0] + '__labels.json'
        d['labels'].append(
            {
                'size': {
                    'x': w, 'y': h
                },
                'label_type': 'box',
                'label_class': label_names[l],
                'object_id': i,
                'centre': {'x': center_x, 'y': center_y}}
        )

    with open(json_filename, 'w') as f:
        json.dump(d, f, skipkeys=True, indent=4)


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument(
        'label_names', help='The path to the yaml file with label names')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='voc0712')
    parser.add_argument('--out')
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

    #vis_bbox(
    #    img, bbox, label, score,
    #    label_names=label_names)
    bbox_to_json(bbox, label, label_names, args.image)
    # if out is not None:
    #     # plot.savefig(args.vis_out)
    # else:
    #     plot.show()


if __name__ == '__main__':
    main()

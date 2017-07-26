import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from chainercv.links import SSD300
from chainercv.visualizations import vis_bbox

from original_detection_dataset import OriginalDetectionDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_dir', help='The root directory of the training dataset')
    parser.add_argument(
        'label_names', help='The path to the yaml file with label names')
    parser.add_argument('--gui', default=False)
    parser.add_argument('--pretrained_model')
    args = parser.parse_args()

    with open(args.label_names, 'r') as f:
        label_names = tuple(yaml.load(f))

    train_data = OriginalDetectionDataset(args.data_dir, label_names)
    
    # Count number of appearence.
    counts = np.zeros((len(label_names),), dtype=np.int32)
    for i in range(len(train_data)):
        img, bbox, label = train_data[i]
        for l in label:
            counts[l] += 1
    for l in range(len(label_names)):
        print('{}   label count {}'.format(
            label_names[l], counts[l]))

    if args.pretrained_model is None:
        return

    model = SSD300(
        n_fg_class=len(label_names),
        pretrained_model=args.pretrained_model)
    model.score_thresh = 0.4

    for i in range(len(train_data)):
        img, gt_bbox, gt_label = train_data[i]
        pred_bbox, pred_label, pred_score = model.predict([img])

        fig = plt.figure(figsize=(32, 16))
        ax1 = fig.add_subplot(1, 2, 1)
        vis_bbox( img, gt_bbox, gt_label, label_names=label_names, ax=ax1)

        ax2 = fig.add_subplot(1, 2, 2)
        vis_bbox(
            img, pred_bbox[0], pred_label[0], pred_score[0],
            label_names=label_names, ax=ax2)
        if not args.gui:
            out_dir = 'analyze_dataset'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            plt.savefig(os.path.join(out_dir, '{}.png'.format(i)))
        else:
            plt.show()


if __name__ == '__main__':
    main()


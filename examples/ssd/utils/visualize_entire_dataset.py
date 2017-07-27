from original_detection_dataset import OriginalDetectionDataset
import yaml

import matplotlib.pyplot as plt
from chainercv.visualizations import vis_bbox


if __name__ == '__main__':
    label_names_fn = '/home/leus/playground/image-labelling-tool/examples/ssd/train/at_home_nagoya_label_names.yml'
    with open(label_names_fn, 'r') as f:
        label_names = tuple(yaml.load(f))

    dataset = OriginalDetectionDataset(
        '/home/leus/data/at_home/', label_names)
    print('length dataset {}'.format(len(dataset)))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(dataset)):
        ax.clear()
        img, bbox, label = dataset[i]
        vis_bbox(img, bbox, label, label_names=label_names, ax=ax)
        plt.savefig('vis_dataset/{:06}'.format(i))

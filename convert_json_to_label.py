import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import yaml

from flask import Flask, render_template, request, make_response, send_from_directory

from image_labelling_tool import labelling_tool


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image labelling tool - Convert Json to Image')
    parser.add_argument('--image_dir')
    parser.add_argument('--label_names')
    parser.add_argument('--file_ext', type=str, default='png')
    args = parser.parse_args()
    file_ext = '.{}'.format(args.file_ext)
    img_dir = args.image_dir
    with open(args.label_names, 'r') as f:
        label_names = yaml.load(f)

    labelled_images = labelling_tool.PersistentLabelledImage.for_directory(
        img_dir, image_filename_pattern='*{}'.format(file_ext))

    for labelled_image in labelled_images:
        if labelled_image.labels is not None:
            labels_2d = labelled_image.render_labels(
                label_classes=label_names,
                pixels_as_vectors=False)
            # Unlabeled should be -1
            labels_2d -= 1
            name = os.path.splitext(labelled_image.image_path)[0]
            np.save(os.path.join(name + '.npy'), labels_2d)

            plt.imshow(labels_2d)
            plt.savefig(os.path.join(name + '_label.png'))
            print('{}  converted'.format(labelled_image.image_path))
        else:
            print('{}  there is no labels'.format(labelled_image.image_path))

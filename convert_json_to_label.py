import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import os

from flask import Flask, render_template, request, make_response, send_from_directory

from image_labelling_tool import labelling_tool


FILE_EXT = '.jpg'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image labelling tool - Convert Json to Image')
    parser.add_argument('--image_dir')
    parser.add_argument('--label_names')
    args = parser.parse_args()
    label_names = args.label_names
    img_dir = args.image_dir

    labelled_images = labelling_tool.PersistentLabelledImage.for_directory(
        img_dir, image_filename_pattern='*.jpg')

    for labelled_image in labelled_images:
        if labelled_image.labels is not None:
            labels_2d = labelled_image.render_labels(
                label_classes=['tree', 'building', 'lake'],
                pixels_as_vectors=False)
            name = os.path.splitext(labelled_image.image_path)[0]
            np.save(os.path.join(name + '.npy'), labels_2d)

            plt.imshow(labels_2d)
            plt.savefig(os.path.join(name + '_label'))
            print('{}  converted'.format(labelled_image.image_path))
        else:
            print('{}  there is no labels'.format(labelled_image.image_path))

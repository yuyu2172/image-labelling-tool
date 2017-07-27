import json
import numpy as np
import os

import chainer
from chainercv.utils import read_image


class OriginalDetectionDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir, label_names):
        self.data_dir = data_dir
        self.label_names = label_names

        self.img_filenames = []
        self.anno_filenames = []
        # for name in sorted(os.listdir(data_dir)):
        for root, dirs, files in os.walk(data_dir):
            for name in sorted(files):
                # If the file is not an image, ignore the file.
                if os.path.splitext(name)[1] != '.jpg':
                    continue
                img_filename = os.path.join(root, name)
                anno_filename = os.path.splitext(img_filename)[0] + '__labels.json'
                if not os.path.exists(anno_filename):
                    continue
                self.img_filenames.append(img_filename)
                self.anno_filenames.append(anno_filename)

    def __len__(self):
        return len(self.img_filenames)

    def get_example(self, i):
        img_filename = self.img_filenames[i]
        anno_filename = self.anno_filenames[i]
        img = read_image(img_filename)

        with open(anno_filename, 'r') as f:
            anno = json.load(f)
        anno = anno['labels']

        bbox = []
        label = []
        for anno_i in anno:
            h = anno_i['size']['y']
            w = anno_i['size']['x']
            center_y = anno_i['centre']['y']
            center_x = anno_i['centre']['x']

            if anno_i['label_class'] not in self.label_names:
                raise ValueError(
                    'The class does not exist {}'.format(anno_i['label_class']))
            l = self.label_names.index(anno_i['label_class'])
            bbox.append(
                [center_y - h / 2, center_x - w / 2,
                 center_y + h / 2, center_x + w / 2])
            label.append(l)

        bbox = np.array(bbox, dtype=np.float32)
        label = np.array(label, dtype=np.int32)
        return img, bbox, label

import argparse
import os
import warnings
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', default='train')
    parser.add_argument('val_dir', default='val')
    parser.add_argument('data_dirs', nargs='*')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    filename_tuples = []
    for data_dir in args.data_dirs:
        for img_name in os.listdir(data_dir):
            if os.path.splitext(img_name)[1] == '.jpg':
                name = os.path.splitext(img_name)[0]
                tup = (os.path.join(data_dir, img_name),
                    os.path.join(data_dir, name + '__labels.json'))
                filename_tuples.append(tup)

    if os.path.exists(args.train_dir):
        warnings.warn('directory train already exists')
    os.makedirs(args.train_dir)

    if os.path.exists(args.val_dir):
        warnings.warn('directory train already exists')
    os.makedirs(args.val_dir)

    order = np.arange(len(filename_tuples))
    np.random.shuffle(order)
    first_size = int(len(order) * 0.8)

    for i in order[:first_size]:
        tup = filename_tuples[i]
        for filename in tup:
            name = os.path.split(filename)[1]
            shutil.copyfile(filename, os.path.join(args.train_dir, name))

    for i in order[first_size:]:
        tup = filename_tuples[i]
        for filename in tup:
            name = os.path.split(filename)[1]
            shutil.copyfile(filename, os.path.join(args.val_dir, name))


if __name__ == '__main__':
    main()

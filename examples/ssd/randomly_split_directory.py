import argparse
import os
import warnings
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', default='train')
    parser.add_argument('val_dir', default='val')
    parser.add_argument('data_dir')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    filename_tuples = []
    for root, dirs, names in os.walk(args.data_dir):
        for name in names:
            if os.path.splitext(name)[1] == '.jpg':
                name_head = os.path.splitext(name)[0]
                tup = (os.path.join(root, name),
                       os.path.join(root, name_head + '__labels.json'),
                       '/'.join(os.path.split(root)[1:]))
                filename_tuples.append(tup)


    if os.path.exists(args.train_dir):
        warnings.warn('directory train already exists')
    else:
        os.makedirs(args.train_dir)

    if os.path.exists(args.val_dir):
        warnings.warn('directory train already exists')
    else:
        os.makedirs(args.val_dir)

    order = np.arange(len(filename_tuples))
    np.random.shuffle(order)
    first_size = int(len(order) * 0.8)

    for i in order[:first_size]:
        tup = filename_tuples[i]
        out_dir = os.path.join(args.train_dir, tup[2])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        name = os.path.split(tup[0])[1]
        shutil.copyfile(tup[0], os.path.join(out_dir, name))
        name = os.path.split(tup[1])[1]
        shutil.copyfile(tup[1], os.path.join(out_dir, name))

    for i in order[first_size:]:
        tup = filename_tuples[i]
        tup = filename_tuples[i]
        out_dir = os.path.join(args.val_dir, tup[2])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        name = os.path.split(tup[0])[1]
        shutil.copyfile(tup[0], os.path.join(out_dir, name))
        name = os.path.split(tup[1])[1]
        shutil.copyfile(tup[1], os.path.join(out_dir, name))


if __name__ == '__main__':
    main()

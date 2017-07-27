import os
import shutil

import numpy as np


if __name__ == '__main__':
    filenames = []
    data_dir = 'data'
    train_dir = 'train'
    val_dir = 'val'

    for img_name in os.listdir(data_dir):
        if os.path.splitext(img_name)[1] == '.jpg':
            name = os.path.splitext(img_name)[0]
            tup = (os.path.join(data_dir, img_name),
                   os.path.join(data_dir, name + '__labels.json'))
            filenames.append(tup)
    
    if os.path.exists(train_dir):
        raise ValueError('directory train already exists')
    os.makedirs(train_dir)

    if os.path.exists(val_dir):
        raise ValueError('directory val already exists')
    os.makedirs(val_dir)

    order = np.arange(len(filenames))
    np.random.shuffle(order)
    first_size = int(len(order) * 0.8)
    
    for i in order[:first_size]:
        tup = filenames[i]
        for filename in tup:
            name = os.path.split(filename)[1]
            shutil.copyfile(filename, os.path.join(train_dir, name))

    for i in order[first_size:]:
        tup = filenames[i]
        for filename in tup:
            name = os.path.split(filename)[1]
            shutil.copyfile(filename, os.path.join(val_dir, name))

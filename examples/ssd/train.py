import argparse
import yaml

import chainer

from original_detection_dataset import OriginalDetectionDataset
from train_utils import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', help='The root directory of the training dataset')
    parser.add_argument(
        '--val',
        help='The root directory of the validation dataset. If this is not '
        'supplied, the data for train dataset is split into two with ratio 8:2.')
    parser.add_argument(
        '--label_names', help='The path to the yaml file with label names')
    parser.add_argument(
        '--iteration', type=int, default=120000,
        help='The number of iterations to run until finishing the train loop')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument(
        '--step_size', type=int, default=-1,
        help='The number of iterations to run before '
        'dropping the learning rate by 0.1')
    parser.add_argument(
        '--batchsize', type=int, default=8,
        help='The size of batch')
    parser.add_argument(
        '--gpu', type=int, default=-1, help='GPU ID')
    parser.add_argument('--out', default='result',
                        help='The directory in which logs are saved')
    parser.add_argument(
        '--val_iteration', type=int, default=10000,
        help='The number of iterations between every validation.')
    parser.add_argument(
        '--log_iteration', type=int, default=10,
        help='The number of iterations between every logging.')
    parser.add_argument(
        '--resume',
        help='The path to the trainer snapshot to resume from. '
        'If unspecified, no snapshot will be resumed')
    args = parser.parse_args()

    with open(args.label_names, 'r') as f:
        label_names = tuple(yaml.load(f))

    if args.val is not None:
        train_data = OriginalDetectionDataset(args.train, label_names)
        val_data = OriginalDetectionDataset(args.val, label_names)
    else:
        # If --val is not supplied, the train data is split into two
        # with ratio 8:2.
        dataset = OriginalDetectionDataset(args.train, label_names)
        train_data, val_data = chainer.datasets.split_dataset_random(
            dataset, int(len(dataset) * 0.8))

    step_points = [args.step_size]
    train(
        train_data,
        val_data,
        label_names,
        args.iteration,
        args.lr,
        step_points,
        args.batchsize,
        args.gpu,
        args.out,
        args.val_iteration,
        args.log_iteration,
        args.resume)

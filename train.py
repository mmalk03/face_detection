#!/usr/bin/env python
import argparse

import matplotlib

from scripts import util

matplotlib.use('agg')

defaults = {
    'base_model': 'vgg16_2',
    'num_classes': 4,
    'epochs': 30,
    'batch_size': 16,
    'train_data_dir': 'data/train',
    'val_data_dir': 'data/validation',
}


def main(args):
    my_network = util.get_network(args.base_model)
    my_network.train(args.train_data_dir, args.val_data_dir, args.epochs, args.batch_size)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Trains given architecture on specified data set.')
    parser.add_argument(
        '--base-model',
        help='Base model architecture',
        type=str,
        default=defaults['base_model'])
    parser.add_argument(
        '--num-classes',
        help='Number of classes in the classification problem.',
        type=int,
        default=defaults['num_classes'])
    parser.add_argument(
        '--epochs',
        help='Number of epochs to train.',
        type=int,
        default=defaults['epochs'])
    parser.add_argument(
        '--batch-size',
        help='Batch size',
        type=int,
        default=defaults['batch_size'])
    parser.add_argument(
        '--train-data-dir',
        help='Directory of training data',
        type=str,
        default=defaults['train_data_dir'])
    parser.add_argument(
        '--val-data-dir',
        help='Directory of validation data',
        type=str,
        default=defaults['val_data_dir'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())

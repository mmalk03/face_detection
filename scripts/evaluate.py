#!/usr/bin/env python
import argparse

import matplotlib

from rappers.scripts import util

matplotlib.use('agg')

defaults = {
    'base_model': 'vgg16',
    'data_dir': 'data/validation'
}


def main(args):
    my_network = util.get_network(args.base_model)
    my_network.evaluate(args.dir)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Trains CNN model using fine-tuned VGG16 architecture with weights from imagenet')
    parser.add_argument(
        '--base-model',
        help='Base model architecture',
        type=str,
        default=defaults['base_model'])
    parser.add_argument(
        '--dir',
        help='Directory with images used for evaluation.',
        type=int,
        default=defaults['data_dir'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())

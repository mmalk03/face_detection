#!/usr/bin/env python
import argparse

import matplotlib

from rappers.scripts import util

matplotlib.use('agg')

defaults = {
    'base_model': 'inceptionv3',
    'data_dir': '../data/validation'
}


def main(args):
    my_network = util.get_network(args.base_model)
    my_network.evaluate(args.dir)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Evaluates given model')
    parser.add_argument(
        '--base-model',
        help='Base model architecture',
        type=str,
        default=defaults['base_model'])
    parser.add_argument(
        '--dir',
        help='Directory with images used for evaluation.',
        type=str,
        default=defaults['data_dir'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())

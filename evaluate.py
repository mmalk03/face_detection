#!/usr/bin/env python
import argparse

import matplotlib

matplotlib.use('agg')

from my_inceptionv3 import MyInceptionV3
from my_resnet50 import MyResNet50
from my_vgg16 import MyVgg16
from my_vgg16_ft import MyVgg16FT

defaults = {
    'base_model': 'vgg16_ft',
    'data_dir': 'data/validation'
}

epochs = 30
batch_size = 16
train_data_dir = 'data/train'
val_data_dir = 'data/validation'


def main(args):
    my_network = {
        'vgg16': MyVgg16(),
        'vgg16_ft': MyVgg16FT(),
        'resnet50': MyResNet50(),
        'inceptionv3': MyInceptionV3()
    }[args.base_model]
    # TODO: implement


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
    parser.add_argument(
        '--num-classes',
        help='Number of classes.',
        type=int,
        default=defaults['num_classes'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())

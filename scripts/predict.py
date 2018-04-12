#!/usr/bin/env python
import argparse

import cv2
import matplotlib

from rappers.scripts import util

matplotlib.use('agg')

defaults = {
    'base_model': 'vgg16_ft',
    'path': '../ostr.png',
    'dir': '../data/validation'
}


def main(args):
    my_network = util.get_network(args.base_model)
    predicted_class = my_network.make_prediction(args.path)

    class_dictionary = util.get_class_dictionary(args.dir)
    in_id = predicted_class[0]
    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[in_id]
    print("Image ID: {}, Label: {}".format(in_id, label))
    original_image = cv2.imread(args.path)
    cv2.putText(original_image, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
    cv2.imshow("Classification", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Predicts class of an image')
    parser.add_argument(
        '--base-model',
        help='Base model architecture',
        type=str,
        default=defaults['base_model'])
    parser.add_argument(
        '--path',
        help='Path to the image.',
        type=str,
        default=defaults['path'])
    parser.add_argument(
        '--dir',
        help='Directory containing images with possible classes.',
        type=str,
        default=defaults['dir'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())

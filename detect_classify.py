#! /usr/bin/env python

import argparse
import json
import os

import cv2
import numpy as np
from keras.models import load_model
from yolo.utils.utils import get_yolo_boxes, makedirs
from yolo.utils.colors import get_color

from scripts import util


def _main_(args):
    config_path = args.conf
    input_path = args.input
    output_path = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    ###############################

    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    # the main loop
    for image_path in image_paths:
        image = cv2.imread(image_path)
        print(image_path)

        # predict the bounding boxes
        boxes = get_yolo_boxes(
            infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh)

        # write the image with bounding boxes to file
        cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))


def get_rapper_name(img_path):
    my_network = util.get_network('vgg16_ft')
    predicted_class = my_network.make_prediction(img_path)

    class_dictionary = util.get_class_dictionary('data/validation')
    in_id = predicted_class[0]
    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[in_id]
    print("Image ID: {}, Label: {}".format(in_id, label))
    return label


def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i
            if not quiet: print(label_str)

        if label >= 0:
            # TODO: cut rectangle and save as in temp_img_path
            # temp_img_path = 'output/temp_img.jpg'
            # rapper_name = get_rapper_name(temp_img_path)
            rapper_name = 'Taco'
            label_str += ' name: ' + rapper_name

            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin - 3, box.ymin],
                               [box.xmin - 3, box.ymin - height - 26],
                               [box.xmin + width + 13, box.ymin - height - 26],
                               [box.xmin + width + 13, box.ymin]], dtype='int32')

            cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=get_color(label),
                          thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin + 13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image.shape[0],
                        color=(0, 0, 0),
                        thickness=2)

    return image


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Detect with a trained yolo model and classify with a trained'
                                                    'VGG16')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image or a directory of images')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')

    args = argparser.parse_args()
    _main_(args)

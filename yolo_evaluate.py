#! /usr/bin/env python

import argparse
import json
import os

from keras.models import load_model

from yolo.ann_parser import parse_csv_annotation
from yolo.generator import BatchGenerator
from yolo.utils.utils import normalize, evaluate


def _main_(args):
    config_path = 'yolo/configs/config_faces_eval.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    if args.data_dir[-1] != '/':
        args.data_dir += '/'
    valid_image_folder = args.data_dir
    valid_annot_file = args.data_dir + 'faces.csv'

    ###############################
    #   Create the validation generator
    ###############################
    valid_ints, labels = parse_csv_annotation(
        valid_annot_file,
        valid_image_folder,
        config['valid']['cache_name']
    )

    labels = labels.keys() if len(config['model']['labels']) == 0 else config['model']['labels']
    labels = sorted(labels)

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=0,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )

    ###############################
    #   Load the model and do evaluation
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    infer_model = load_model(config['train']['saved_weights_name'])

    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-d', '--data-dir', help='path to dataset')

    args = argparser.parse_args()
    _main_(args)

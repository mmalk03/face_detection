#! /usr/bin/env python
import pickle
import argparse
import json
import os

import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from yolo.ann_parser import parse_csv_annotation
from yolo.callbacks import CustomModelCheckpoint, CustomTensorBoard
from yolo.generator import BatchGenerator
from yolo.utils.utils import normalize, evaluate, makedirs
from yolo.yolo import create_yolov3_model, dummy_loss


def create_training_instances(train_annot_file, train_image_folder, train_cache,
                              valid_annot_file, valid_image_folder, valid_cache):
    # parse annotations of the training set
    train_ints, train_labels = parse_csv_annotation(train_annot_file, train_image_folder, train_cache)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_file):
        valid_ints, valid_labels = parse_csv_annotation(valid_annot_file, valid_image_folder, valid_cache)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8 * len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    print('Training on labels: \t' + str(train_labels) + '\n')
    labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save, mAP_path, infer_model, val_generator):
    makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=5,
        mode='min',
        verbose=1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save=model_to_save,
        mAP_path=mAP_path,
        infer_model=infer_model,
        val_generator=val_generator,
        filepath=saved_weights_name,  # + '{epoch:02d}.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        epsilon=0.01,
        cooldown=0,
        min_lr=0
    )
    tensorboard = CustomTensorBoard(
        log_dir=tensorboard_logs,
        write_graph=True,
        write_images=True,
    )
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]


def create_model(nb_class, anchors, max_box_per_image, max_grid, batch_size, warmup_batches, ignore_thresh,
                 saved_weights_name, lr, grid_scales, obj_scale, noobj_scale, xywh_scale, class_scale):
    template_model, infer_model = create_yolov3_model(
        nb_class=nb_class,
        anchors=anchors,
        max_box_per_image=max_box_per_image,
        max_grid=max_grid,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        ignore_thresh=ignore_thresh,
        grid_scales=grid_scales,
        obj_scale=obj_scale,
        noobj_scale=noobj_scale,
        xywh_scale=xywh_scale,
        class_scale=class_scale
    )

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights("backend.h5", by_name=True)

    train_model = template_model

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


def _main_(args):
    config_path = args.conf
    print(config_path)
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_file'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_file'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators 
    ###############################
    train_generator = BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.3,
        norm=normalize
    )

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )

    ###############################
    #   Create the model 
    ###############################
    if os.path.exists(config['train']['saved_weights_name']):
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']

    train_model, infer_model = create_model(
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh'],
        saved_weights_name=config['train']['saved_weights_name'],
        lr=config['train']['learning_rate'],
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model,
                                 config['train']['mAP_path'], infer_model, valid_generator)

    history = train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=4,
        max_queue_size=8
    )

    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    save_history(config['train']['saved_history_name'], history.history)


def save_history(path, history):
    with open(path, 'wb') as file_pi:
        pickle.dump(history, file_pi)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)

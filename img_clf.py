#!/usr/bin/env python
import argparse
import itertools

import cv2
import matplotlib

from rappers import util

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

from my_inceptionv3 import MyInceptionV3
from my_resnet50 import MyResNet50
from my_vgg16 import MyVgg16
from my_resnet50_finetuned import MyResNet50FineTuned
from my_vgg16_ft import MyVgg16FT

defaults = {
    'base_model': 'vgg16_ft',
    'num_classes': 4,
    'epochs': 10,
    'batch_size': 16,
    'train_data_dir': 'data/train',
    'val_data_dir': 'data/validation',
}

img_width = 224
img_height = 224
incv3_img_width = 299
incv3_img_height = 299


def main(args):
    my_network = {
        'vgg16': MyVgg16(img_width, img_height, args.num_classes, args.epochs, args.batch_size,
                         args.train_data_dir, args.val_data_dir),
        'vgg16_ft': MyVgg16FT(img_width, img_height, args.num_classes, args.epochs, args.batch_size,
                              args.train_data_dir, args.val_data_dir),
        'resnet50': MyResNet50(img_width, img_height, args.num_classes, args.epochs, args.batch_size,
                               args.train_data_dir, args.val_data_dir),
        'resnet50_ft': MyResNet50FineTuned(img_width, img_height, args.num_classes, args.epochs, args.batch_size,
                                           args.train_data_dir, args.val_data_dir),
        'inceptionv3': MyInceptionV3(incv3_img_width, incv3_img_height, args.num_classes, args.epochs, args.batch_size,
                                     args.train_data_dir, args.val_data_dir)
    }[args.base_model]

    train_network(my_network)
    # make_prediction(args, my_network, 'ostr.png')
    # validate_accuracy(my_network)
    # compare_all_networks(args, train_generator, val_generator)
    # plot_confusion_matrix(my_network, val_generator)
    # make_filter_vis(args, my_network)


def train_network(network):
    history = network.train()
    plot_history(history)


def compare_all_networks(args):
    my_vgg = MyVgg16(img_width, img_height, args.num_classes, args.epochs, args.batch_size,
                     args.train_data_dir, args.val_data_dir)
    my_resnet = MyResNet50(incv3_img_width, incv3_img_height, args.num_classes, args.epochs, args.batch_size,
                           args.train_data_dir, args.val_data_dir)
    my_inception = MyInceptionV3(img_width, img_height, args.num_classes, args.epochs, args.batch_size,
                                 args.train_data_dir, args.val_data_dir)
    vgg_hist = my_vgg.train()
    resnet_hist = my_resnet.train()
    inception_hist = my_inception.train()
    compare_all_histories(vgg_hist, resnet_hist, inception_hist)


def load_image(image_path, width, height):
    image = load_img(image_path, target_size=(width, height))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return image


def make_prediction(args, network, image_path):
    if args.base_model == 'inceptionv3':
        image = load_image(image_path, incv3_img_width, incv3_img_height)
    else:
        image = load_image(image_path, img_width, img_height)
    class_dictionary = util.get_class_dictionary(args.val_data_dir)
    predicted_class = network.make_prediction(image)

    in_id = predicted_class[0]
    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[in_id]
    print("Image ID: {}, Label: {}".format(in_id, label))
    original_image = cv2.imread(image_path)
    cv2.putText(original_image, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
    cv2.imshow("Classification", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_history(history):
    plt.figure(1)

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def compare_all_histories(vgg_hist, resnet_hist, inception_hist):
    plt.figure(1)

    plt.subplot(221)
    plt.plot(vgg_hist.history['acc'])
    plt.plot(resnet_hist.history['acc'])
    plt.plot(inception_hist.history['acc'])
    plt.title('Accuracy')
    plt.ylabel('Training')
    plt.legend(['vgg', 'resnet', 'inception'], loc='lower right')

    plt.subplot(223)
    plt.plot(vgg_hist.history['val_acc'])
    plt.plot(resnet_hist.history['val_acc'])
    plt.plot(inception_hist.history['val_acc'])
    plt.ylabel('Validation')
    plt.xlabel('epoch')
    plt.legend(['vgg', 'resnet', 'inception'], loc='lower right')

    plt.subplot(222)
    plt.plot(vgg_hist.history['loss'])
    plt.plot(resnet_hist.history['loss'])
    plt.plot(inception_hist.history['loss'])
    plt.title('Loss')
    plt.legend(['vgg', 'resnet', 'inception'], loc='upper right')

    plt.subplot(224)
    plt.plot(vgg_hist.history['val_loss'])
    plt.plot(resnet_hist.history['val_loss'])
    plt.plot(inception_hist.history['val_loss'])
    plt.xlabel('epoch')
    plt.legend(['vgg', 'resnet', 'inception'], loc='upper right')

    plt.tight_layout()
    plt.savefig('plots/network_compare.png')
    plt.show()


def plot_confusion_matrix(args, network, normalize=False, title='Confusion matrix'):
    conf_mat = network.get_confusion_matrix(args.val_data_dir)
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(conf_mat)

    class_dictionary = util.get_class_dictionary(args.val_data_dir)

    np.set_printoptions(precision=2)
    plt.figure()
    tick_marks = np.arange(len(class_dictionary))
    plt.title(title)
    plt.imshow(conf_mat, interpolation='none', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt.xticks(tick_marks, class_dictionary.keys(), rotation=20, fontsize=10)
    plt.yticks(tick_marks, class_dictionary.keys(), fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('plots.confusion_matrix.png')
    plt.show()


def make_filter_vis(args, network):
    if args.base_model == 'vgg16':
        network.visualize_filters()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Trains CNN model using fine-tuned VGG16 architecture with weights from imagenet')
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

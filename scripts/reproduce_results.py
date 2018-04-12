#!/usr/bin/env python
import argparse
import itertools
import os

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from random import randint

from rappers.networks.my_vgg16 import MyVgg16
from rappers.networks.my_vgg16_ft import MyVgg16FT
from rappers.scripts import util

matplotlib.use('agg')

defaults = {
    'dir': '../data/validation'
}


def main(args):
    plot_save_dir = '../report/plots/'

    my_vgg16 = MyVgg16()
    my_vgg16_ft = MyVgg16FT()

    vgg16_hist = util.load_history('../report/histories/vgg16_history')
    vgg16_da_hist = util.load_history('../report/histories/vgg16_da_history')
    vgg16_ft_hist = util.load_history('../report/histories/vgg16_ft_history')

    vgg16_adam_hist = util.load_history('../report/histories/vgg16_adam_history')
    vgg16_bs32_hist = util.load_history('../report/histories/vgg16_bs32_history')
    vgg16_rmsprop_1e7_hist = util.load_history('../report/histories/vgg16_rmsprop_1e7_history')
    vgg16_rmsprop_001_hist = util.load_history('../report/histories/vgg16_rmsprop_001_history')

    resnet50_hist = util.load_history('../report/histories/resnet50_30e_history')
    resnet50_da02_hist = util.load_history('../report/histories/resnet50_30e_da02_history')
    resnet50__100e_da02_hist = util.load_history('../report/histories/resnet50_100e_da02_history')

    inceptionv3_hist = util.load_history('../report/histories/inceptionv3_history')

    compare_base_model_histories(vgg16_hist, resnet50_hist, inceptionv3_hist)
    compare_2_histories(vgg16_hist, vgg16_da_hist,
                        'VGG16', 'VGG16, with data augmentation',
                        plot_save_dir + 'vgg16_da_compare_acc.png',
                        plot_save_dir + 'vgg16_da_compare_loss.png')
    compare_2_histories(vgg16_hist, vgg16_ft_hist,
                        'VGG16', 'VGG16, with fine-tuning',
                        plot_save_dir + 'vgg16_ft_compare_acc.png',
                        plot_save_dir + 'vgg16_ft_compare_loss.png')
    compare_2_histories(vgg16_hist, vgg16_bs32_hist,
                        'VGG16, batch_size=16', 'VGG16, batch_size=32',
                        plot_save_dir + 'vgg16_bs_compare_acc.png',
                        plot_save_dir + 'vgg16_bs_compare_loss.png')
    compare_2_histories(vgg16_hist, vgg16_adam_hist,
                        'VGG16, default RMSprop optimizer', 'VGG16, default Adam optimizer',
                        plot_save_dir + 'vgg16_rms_adam_compare_acc.png',
                        plot_save_dir + 'vgg16_rms_adam_compare_loss.png')
    compare_2_histories(vgg16_hist, vgg16_rmsprop_1e7_hist,
                        'VGG16, default RMSprop optimizer', 'VGG16, RMSprop with learning rate = 1e-7',
                        plot_save_dir + 'vgg16_rms_small_lr_compare_acc.png',
                        plot_save_dir + 'vgg16_rms_small_lr_compare_loss.png')
    compare_2_histories(vgg16_hist, vgg16_rmsprop_001_hist,
                        'VGG16, default RMSprop optimizer', 'VGG16, RMSprop with learning rate = 0.01',
                        plot_save_dir + 'vgg16_rms_big_lr_compare_acc.png',
                        plot_save_dir + 'vgg16_rms_big_lr_compare_loss.png')
    compare_2_histories(resnet50_hist, resnet50_da02_hist,
                        'ResNet50, without dropout', 'ResNet50, with dropout = 0.2',
                        plot_save_dir + 'resnet50_dropout_compare_acc.png',
                        plot_save_dir + 'resnet50_dropout_compare_loss.png')
    save_plot_history(resnet50__100e_da02_hist,
                      'ResNet50, with dropout = 0.2',
                      plot_save_dir + 'resnet50_100e_da02_acc.png',
                      plot_save_dir + 'resnet50_100e_da02_loss.png')

    plot_confusion_matrix(my_vgg16_ft, args.dir)
    my_vgg16.visualize_filters()
    show_incorrect_predictions(my_vgg16_ft, args.dir)


def load_image(image_path, width, height):
    image = load_img(image_path, target_size=(width, height))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return image


def save_plot_history(hist, label, path_acc, path_loss):
    lt = label + ' train'
    lv = label + ' validation'

    plt.figure()
    plt.plot(hist['acc'], label=lt, color='firebrick')
    plt.plot(hist['val_acc'], label=lv, color='orangered')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path_acc)

    plt.figure()
    plt.plot(hist['loss'], label=lt, color='firebrick')
    plt.plot(hist['val_loss'], label=lv, color='orangered')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_loss)


def compare_base_model_histories(vgg_hist, resnet_hist, inception_hist):
    plt.figure()
    plt.plot(vgg_hist['acc'], label='vgg train', color='firebrick')
    plt.plot(vgg_hist['val_acc'], label='vgg validation', color='orangered')
    plt.plot(resnet_hist['acc'], label='resnet train', color='forestgreen')
    plt.plot(resnet_hist['val_acc'], label='resnet validation', color='springgreen')
    plt.plot(inception_hist['acc'], label='inception train', color='steelblue')
    plt.plot(inception_hist['val_acc'], label='inception validation', color='paleturquoise')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('../report/plots/base_model_compare_acc.png')

    plt.figure()
    plt.plot(vgg_hist['loss'], label='vgg train', color='firebrick')
    plt.plot(vgg_hist['val_loss'], label='vgg validation', color='orangered')
    plt.plot(resnet_hist['loss'], label='resnet train', color='forestgreen')
    plt.plot(resnet_hist['val_loss'], label='resnet validation', color='springgreen')
    plt.plot(inception_hist['loss'], label='inception train', color='steelblue')
    plt.plot(inception_hist['val_loss'], label='inception validation', color='paleturquoise')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('../report/plots/base_model_compare_loss.png')


def compare_normal_with_ft_histories(hist, ft_hist):
    plt.figure()
    plt.plot(hist['acc'], label='Normal train', color='firebrick')
    plt.plot(hist['val_acc'], label='Normal validation', color='orangered')
    plt.plot(ft_hist['acc'], label='Fine-tuned train', color='springgreen')
    plt.plot(ft_hist['val_acc'], label='Fine-tuned validation', color='forestgreen')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('../report/plots/standard_ft_compare_acc.png')

    plt.figure()
    plt.plot(hist['loss'], label='Normal train', color='firebrick')
    plt.plot(hist['val_loss'], label='Normal validation', color='orangered')
    plt.plot(ft_hist['loss'], label='Fine-tuned train', color='springgreen')
    plt.plot(ft_hist['val_loss'], label='Fine-tuned validation', color='forestgreen')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('../report/plots/standard_ft_compare_loss.png')


def compare_2_histories(hist1, hist2, label1, label2, path_acc, path_loss):
    l1t = label1 + ' train'
    l1v = label1 + ' validation'
    l2t = label2 + ' train'
    l2v = label2 + ' validation'

    plt.figure()
    plt.plot(hist1['acc'], label=l1t, color='firebrick')
    plt.plot(hist1['val_acc'], label=l1v, color='orangered')
    plt.plot(hist2['acc'], label=l2t, color='springgreen')
    plt.plot(hist2['val_acc'], label=l2v, color='forestgreen')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path_acc)

    plt.figure()
    plt.plot(hist1['loss'], label=l1t, color='firebrick')
    plt.plot(hist1['val_loss'], label=l1v, color='orangered')
    plt.plot(hist2['loss'], label=l2t, color='springgreen')
    plt.plot(hist2['val_loss'], label=l2v, color='forestgreen')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_loss)


def plot_confusion_matrix(network, data_dir, normalize=False, title='Confusion matrix'):
    conf_mat = network.get_confusion_matrix(data_dir)
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(conf_mat)

    class_dictionary = util.get_class_dictionary(data_dir)

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
    plt.savefig('../report/plots/confusion_matrix.png')
    plt.show()


def show_incorrect_predictions(network, data_dir):
    images = []
    for root, directories, filenames in os.walk(data_dir, topdown=True):
        directories.sort(reverse=False)
        for filename in filenames:
            images.append(os.path.join(root, filename))

    class_dictionary = util.get_class_dictionary(data_dir)
    inc = network.get_wrong_predictions(data_dir)
    incorrects = inc[0]

    for i in range(4):
        random_image_index = randint(0, len(incorrects))
        img_index = incorrects[random_image_index]
        random_wrong_image = images[img_index]
        predicted_class = network.make_prediction(random_wrong_image)

        pred_label_index = predicted_class[0]
        true_label_index = (img_index // 400)
        # TODO fix this
        print(class_dictionary.items())
        for (k, v) in class_dictionary.items():
            print(k, ' ', v)

        pred_label = [k for (k, v) in class_dictionary.items() if v == pred_label_index]
        true_label = [k for (k, v) in class_dictionary.items() if v == true_label_index]
        print('Predicted: ', pred_label)
        print('Actual: ', true_label)
        print('Path: ', random_wrong_image)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Reproduce results')
    parser.add_argument(
        '--dir',
        help='Directory with images belonging to considered classes.',
        type=str,
        default=defaults['dir'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())

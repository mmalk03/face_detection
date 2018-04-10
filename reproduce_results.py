#!/usr/bin/env python
import argparse
import itertools

import matplotlib

from rappers import util

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

from my_inceptionv3 import MyInceptionV3
from my_resnet50 import MyResNet50
from my_vgg16 import MyVgg16
from my_vgg16_ft import MyVgg16FT

defaults = {
    'dir': 'data/validation'
}


def main(args):
    my_vgg16 = MyVgg16()
    my_vgg16_ft = MyVgg16FT()
    my_resnet50 = MyResNet50()
    my_inceptionv3 = MyInceptionV3()

    vgg16_hist = my_vgg16.get_history()
    resnet50_hist = my_resnet50.get_history()
    inceptionv3_hist = my_inceptionv3.get_history()
    compare_all_histories(vgg16_hist, resnet50_hist, inceptionv3_hist)

    vgg16_ft_hist = my_vgg16_ft.get_history()
    plot_history(vgg16_ft_hist)

    plot_confusion_matrix(my_vgg16_ft, args.dir)

    my_vgg16.visualize_filters()


def load_image(image_path, width, height):
    image = load_img(image_path, target_size=(width, height))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return image


def plot_history(history):
    plt.figure(1)

    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('plots/network_compare.png')
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


def plot_confusion_matrix(network, dir, normalize=False, title='Confusion matrix'):
    conf_mat = network.get_confusion_matrix(dir)
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(conf_mat)

    class_dictionary = util.get_class_dictionary(dir)

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

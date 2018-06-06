#!/usr/bin/env python
import pickle

import matplotlib.pyplot as plt
import numpy as np

plot_save_dir = 'report/plots/'
history_dir = 'histories/'


def main():
    hist1_name = 'face_01_06_15-25'
    hist2_name = 'face_05_06_15-08'
    hist3_name = 'face_06_06_20-24'
    process_history(hist1_name)
    process_history(hist2_name)
    process_history(hist3_name)
    compare_2_histories(hist1_name, hist2_name, '288 - 488, bs=2', '128 - 288, bs=4')


def process_history(hist_name):
    hist = load_history(history_dir + hist_name)
    save_loss_history(hist, plot_save_dir + 'yolo_hist_loss_' + hist_name + '.png')
    save_map_history(history_dir + hist_name + '_mAP.txt', plot_save_dir + 'yolo_hist_map_' + hist_name + '.png')


def load_history(path):
    filename = open(path, 'rb')
    history = pickle.load(filename)
    filename.close()
    return history


def save_loss_history(hist, path):
    l = 'loss'
    l1 = 'yolo layer 1 loss'
    l2 = 'yolo layer 2 loss'
    l3 = 'yolo layer 3 loss'

    plt.figure()
    x = np.arange(-2, len(hist['loss']) - 2)
    plt.plot(x, hist['loss'], label=l, color='lightcoral')
    plt.plot(x, hist['yolo_layer_1_loss'], label=l1, color='cadetblue')
    plt.plot(x, hist['yolo_layer_2_loss'], label=l2, color='darkslategray')
    plt.plot(x, hist['yolo_layer_3_loss'], label=l3, color='crimson')

    for var in (hist['loss'], hist['yolo_layer_1_loss'], hist['yolo_layer_2_loss'], hist['yolo_layer_3_loss']):
        plt.annotate('%0.3f' % var[-1], xy=(1, var[-1]), xytext=(8, 0), va='center',
                     xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.xlabel('Epoch')
    plt.ylim(ymax=10, ymin=0)
    plt.axvline(x=0, label='End of warm up training', linestyle='dashed')
    plt.legend(loc='upper right')
    plt.savefig(path)


def save_map_history(map_path, save_path):
    l = 'Mean Average Precision at 0.5'
    with open(map_path) as f:
        numbers = f.read()
        str_list = numbers.split('\n')
        str_list.pop()
        float_list = [float(x) for x in str_list]
        x = np.arange(-2, len(float_list) - 2)
        plt.figure()
        plt.plot(x, float_list, label=l, color='lightcoral')
        plt.annotate('%0.3f' % float_list[-1], xy=(1, float_list[-1]), xytext=(8, 0), va='center',
                     xycoords=('axes fraction', 'data'), textcoords='offset points')
        plt.xlabel('Epoch')
        plt.axvline(x=0, label='End of warm up training', linestyle='dashed')
        plt.legend(loc='lower right')
        plt.savefig(save_path)


def compare_2_histories(hist_name1, hist_name2, label1, label2):
    hist1 = load_history(history_dir + hist_name1)
    hist2 = load_history(history_dir + hist_name2)
    path_map1 = history_dir + hist_name1 + '_mAP.txt'
    path_map2 = history_dir + hist_name2 + '_mAP.txt'
    path_loss = plot_save_dir + 'compare_loss_' + hist_name1 + '-' + hist_name2 + '.png'
    map_save_path = plot_save_dir + 'compare_map_' + hist_name1 + '-' + hist_name2 + '.png'

    h1_l = label1 + ' loss'
    h1_l1 = label1 + ' yolo layer 1 loss'
    h1_l2 = label1 + ' yolo layer 2 loss'
    h1_l3 = label1 + ' yolo layer 3 loss'
    h2_l = label2 + ' loss'
    h2_l1 = label2 + ' yolo layer 1 loss'
    h2_l2 = label2 + ' yolo layer 2 loss'
    h2_l3 = label2 + ' yolo layer 3 loss'

    n = min(len(hist1['loss']), len(hist2['loss']))
    x = np.arange(-2, n - 2)

    plt.figure()
    plt.plot(x, hist1['loss'][:n], label=h1_l, color='#f57c00', linestyle='-.')
    plt.plot(x, hist1['yolo_layer_1_loss'][:n], label=h1_l1, color='#ffc947', linestyle='-.')
    plt.plot(x, hist1['yolo_layer_2_loss'][:n], label=h1_l2, color='#ff9800', linestyle='-.')
    plt.plot(x, hist1['yolo_layer_3_loss'][:n], label=h1_l3, color='#c66900', linestyle='-.')
    plt.plot(x, hist2['loss'][:n], label=h2_l, color='#0288d1')
    plt.plot(x, hist2['yolo_layer_1_loss'][:n], label=h2_l1, color='#67daff')
    plt.plot(x, hist2['yolo_layer_2_loss'][:n], label=h2_l2, color='#03a9f4')
    plt.plot(x, hist2['yolo_layer_3_loss'][:n], label=h2_l3, color='#007ac1')

    # for var in (
    #         hist1['loss'][:n], hist1['yolo_layer_1_loss'][:n],
    #         hist1['yolo_layer_2_loss'][:n], hist1['yolo_layer_3_loss'][:n],
    #         hist2['loss'][:n], hist2['yolo_layer_1_loss'][:n],
    #         hist2['yolo_layer_2_loss'][:n], hist2['yolo_layer_3_loss'][:n]):
    #     plt.annotate('%0.3f' % var[-1], xy=(1, var[-1]), xytext=(8, 0), va='center',
    #                  xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.xlabel('Epoch')
    plt.ylim(ymax=12, ymin=0)
    plt.xlim(xmin=-3)
    plt.axvline(x=0, label='End of warm up training', linestyle='dotted')
    plt.legend(loc='upper right')
    plt.savefig(path_loss)

    h1_l = label1 + ' Mean Average Precision at 0.5'
    h2_l = label2 + ' Mean Average Precision at 0.5'
    with open(path_map1) as f1:
        with open(path_map2) as f2:
            numbers1 = f1.read()
            str_list1 = numbers1.split('\n')
            str_list1.pop()
            float_list1 = [float(x) for x in str_list1]

            numbers2 = f2.read()
            str_list2 = numbers2.split('\n')
            str_list2.pop()
            float_list2 = [float(x) for x in str_list2]

            plt.figure()
            plt.plot(x, float_list1[:n], label=h1_l)
            plt.plot(x, float_list2[:n], label=h2_l)

            # for var in (float_list1[:n], float_list2[:n]):
            #     plt.annotate('%0.3f' % var[-1], xy=(1, var[-1]), xytext=(8, 0), va='center',
            #                  xycoords=('axes fraction', 'data'), textcoords='offset points')

            plt.xlabel('Epoch')
            plt.axvline(x=0, label='End of warm up training', linestyle='dotted')
            plt.legend(loc='lower right')
            plt.savefig(map_save_path)


if __name__ == '__main__':
    main()

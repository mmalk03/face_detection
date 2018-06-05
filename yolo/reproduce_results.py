#!/usr/bin/env python
import pickle

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.use('agg')

plot_save_dir = 'report/plots/'
history_dir = 'histories/'


def main():
    process('face_01_06_15-25')


def process(hist_name):
    hist = load_history(history_dir + hist_name)
    save_loss_history(hist, plot_save_dir + 'yolo_hist_loss_' + hist_name + '.png')
    save_map_history(history_dir + hist_name + '_mAP.txt', plot_save_dir + 'yolo_hist_map_' + hist_name + '.png')


def load_history(path):
    filename = open(path, "rb")
    history = pickle.load(filename)
    filename.close()
    return history


def save_loss_history(hist, path):
    l = 'loss'
    l1 = 'yolo layer 1 loss'
    l2 = 'yolo layer 2 loss'
    l3 = 'yolo layer 3 loss'

    plt.figure()
    plt.plot(hist['loss'], label=l, color='lightcoral')
    plt.plot(hist['yolo_layer_1_loss'], label=l1, color='cadetblue')
    plt.plot(hist['yolo_layer_2_loss'], label=l2, color='darkslategray')
    plt.plot(hist['yolo_layer_3_loss'], label=l3, color='crimson')
    plt.xlabel('Epoch')
    plt.ylim(ymax=12, ymin=0)
    plt.axvline(x=2, label='End of warm up training')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path)


def save_map_history(map_path, save_path):
    l = 'Mean Average Precision at 0.5'
    with open(map_path) as f:
        numbers = f.read()
        str_list = numbers.split('\n')
        str_list.pop()
        float_list = [float(x) for x in str_list]
        plt.figure()
        plt.plot(float_list, label=l, color='lightcoral')
        plt.xlabel('Epoch')
        plt.axvline(x=2, label='End of warm up training')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path)


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


if __name__ == '__main__':
    main()

import csv
import os
import pickle

from PIL import Image


def parse_csv_annotation(ann_file_path, img_dir, cache_name, labels=[]):
    class_label = 'face'
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        with open(ann_file_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader)
            for row in spamreader:
                img_path = img_dir + row[0]
                ymin = int(row[1])
                xmax = int(row[2])
                ymax = int(row[3])
                xmin = int(row[4])

                try:
                    image = Image.open(img_path)
                except FileNotFoundError:
                    print('Couldn\'t open file: ' + img_path + ' - file not found')
                    continue
                width, height = image.size

                img_already_has_object = False
                for im in all_insts:
                    if im['filename'] == img_path:
                        img = im
                        img_already_has_object = True
                        break

                if not img_already_has_object:
                    img = {'object': [], 'filename': img_path, 'width': width, 'height': height}

                obj = {'name': class_label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}

                if obj['name'] in seen_labels:
                    seen_labels[obj['name']] += 1
                else:
                    seen_labels[obj['name']] = 1

                img['object'] += [obj]
                if len(img['object']) > 0 and not img_already_has_object:
                    all_insts += [img]

    return all_insts, seen_labels

import pickle
from keras.preprocessing.image import ImageDataGenerator

from networks.my_inceptionv3 import MyInceptionV3
from networks.my_resnet50 import MyResNet50
from networks.my_resnet50_ft import MyResNet50FT
from networks.my_vgg16 import MyVgg16
from networks.my_vgg16_2 import MyVgg162
from networks.my_vgg16_data_aug import MyVgg16DA
from networks.my_vgg16_ft import MyVgg16FT


def get_class_dictionary(directory):
    return ImageDataGenerator().flow_from_directory(directory).class_indices


def get_network(model_name):
    return {
        'vgg16': MyVgg16(),
        'vgg16_2': MyVgg162(),
        'vgg16_da': MyVgg16DA(),
        'vgg16_ft': MyVgg16FT(),
        'resnet50': MyResNet50(),
        'resnet50_ft': MyResNet50FT(),
        'inceptionv3': MyInceptionV3()
    }[model_name]


def load_history(path):
    filename = open(path, "rb")
    history = pickle.load(filename)
    filename.close()
    return history

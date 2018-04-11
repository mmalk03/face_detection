from keras.preprocessing.image import ImageDataGenerator

from rappers.networks.my_inceptionv3 import MyInceptionV3
from rappers.networks.my_resnet50 import MyResNet50
from rappers.networks.my_vgg16 import MyVgg16
from rappers.networks.my_vgg16_ft import MyVgg16FT


def get_class_dictionary(directory):
    return ImageDataGenerator().flow_from_directory(directory).class_indices


def get_network(model_name):
    return {
        'vgg16': MyVgg16(),
        'vgg16_ft': MyVgg16FT(),
        'resnet50': MyResNet50(),
        'inceptionv3': MyInceptionV3()
    }[model_name]

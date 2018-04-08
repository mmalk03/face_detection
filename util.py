from keras.preprocessing.image import ImageDataGenerator


def get_bottleneck_generator(directory, img_width, img_height, batch_size):
    data_generator = ImageDataGenerator(rescale=1. / 255)
    return data_generator.flow_from_directory(
        directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)


def get_normal_generator(directory, img_width, img_height, batch_size):
    data_generator = ImageDataGenerator(rescale=1. / 255)
    return data_generator.flow_from_directory(
        directory,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

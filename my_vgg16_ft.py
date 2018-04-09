from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

from rappers import util


class MyVgg16FT:
    vgg16_ft_model_plot_path = 'plots/vgg16_ft_model_plot.png'
    top_model_plot_path = 'plots/vgg16_ft_top_model_plot.png'
    train_bottleneck_features_path = 'weights/vgg16_ft_bottleneck_features_train.npy'
    val_bottleneck_features_path = 'weights/vgg16_ft_bottleneck_features_validation.npy'
    top_model_weights_path = 'weights/vgg16_ft_bottleneck_fc_model.h5'
    model_weights_path = 'weights/vgg16_ft_weights.h5'
    history_path = 'histories/vgg16_ft_history'

    img_width = 0
    img_height = 0
    num_classes = 0
    epochs = 0
    batch_size = 0
    train_data_dir = ''
    validation_data_dir = ''

    def __init__(self, img_width, img_height, num_classes, epochs, batch_size, train_data_dir, validation_data_dir):
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir

    def train(self):
        # self.save_bottleneck_features()
        # self.train_top_model()
        return self.fine_tune()

    def save_bottleneck_features(self):
        train_generator = util.get_generator(
            self.train_data_dir,
            self.img_width,
            self.img_height,
            self.batch_size)
        val_generator = util.get_generator(
            self.validation_data_dir,
            self.img_width,
            self.img_height,
            self.batch_size)

        model = self.get_base_model()
        util.save_model_plot(self.vgg16_ft_model_plot_path, model)

        train_bottleneck_features = model.predict_generator(
            train_generator, len(train_generator.filenames) // self.batch_size)
        util.save_bottleneck_features(self.train_bottleneck_features_path, train_bottleneck_features)

        val_bottleneck_features = model.predict_generator(
            val_generator, len(val_generator.filenames) // self.batch_size)
        util.save_bottleneck_features(self.val_bottleneck_features_path, val_bottleneck_features)

    def train_top_model(self):
        train_generator = util.get_generator(
            self.train_data_dir,
            self.img_width,
            self.img_height,
            self.batch_size)
        val_generator = util.get_generator(
            self.validation_data_dir,
            self.img_width,
            self.img_height,
            self.batch_size)

        train_data = util.load_bottleneck_features(self.train_bottleneck_features_path)
        val_data = util.load_bottleneck_features(self.val_bottleneck_features_path)

        train_labels = train_generator.classes
        train_labels = to_categorical(train_labels, num_classes=self.num_classes)
        val_labels = val_generator.classes
        val_labels = to_categorical(val_labels, num_classes=self.num_classes)

        model = self.get_top_model(train_data.shape[1:])
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        util.save_model_plot(self.top_model_plot_path, model)

        history = model.fit(train_data, train_labels,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(val_data, val_labels))
        model.save_weights(self.top_model_weights_path)
        util.eval_model_loss_acc(model, val_data, val_labels, self.batch_size)
        return history

    def fine_tune(self):
        train_generator = util.get_categorical_generator(
            self.train_data_dir,
            self.img_width,
            self.img_height,
            self.batch_size)
        val_generator = util.get_categorical_generator(
            self.validation_data_dir,
            self.img_width,
            self.img_height,
            self.batch_size)

        base_model = self.get_base_model()
        base_model.trainable = True
        set_trainable = False
        for layer in base_model.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        top_model = self.get_top_model(base_model.output_shape[1:])
        top_model.load_weights(self.top_model_weights_path)
        top_model.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(top_model)
        model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        util.save_model_plot(self.vgg16_ft_model_plot_path, model)

        # TODO: extract 6400 and 1600 from generator

        history = model.fit_generator(train_generator,
                                      verbose=1,
                                      steps_per_epoch=6400 / self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=val_generator,
                                      validation_steps=1600 / self.batch_size)
        model.save_weights(self.model_weights_path)
        util.save_history(self.history_path, history)
        return history

    def make_prediction(self, image):
        base_model = self.get_base_model()
        top_model = self.get_top_model(base_model.output_shape[1:])
        model = Sequential()
        model.add(base_model)
        model.add(top_model)
        util.load_model_weights(self.model_weights_path, model)
        # TODO: verify predict classes
        return model.predict_classes(image)

    def get_confusion_matrix(self, directory):
        base_model = self.get_base_model()
        top_model = self.get_top_model(base_model.output_shape[1:])
        model = Sequential()
        model.add(base_model)
        model.add(top_model)
        util.load_model_weights(self.model_weights_path, model)

        generator = util.get_generator(
            directory,
            self.img_width,
            self.img_height,
            self.batch_size)
        train_labels = generator.classes
        # TODO: verify predict classes
        predicted_labels = top_model.predict_classes(generator)
        return confusion_matrix(train_labels, predicted_labels)

    def get_top_model(self, input_shape):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def get_base_model(self):
        return applications.VGG16(include_top=False, weights='imagenet',
                                  input_shape=(self.img_width, self.img_height, 3))

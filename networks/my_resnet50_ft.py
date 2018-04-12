import numpy as np
from keras.applications import ResNet50
from keras.layers import Flatten, Dense
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

from networks import util


class MyResNet50FT:
    base_model_plot_path = 'plots/resnet50_ft_model_plot.png'
    top_model_plot_path = 'plots/resnet50_ft_top_model_plot.png'
    model_plot_path = 'plots/resnet50_ft_model_plot.png'
    train_bottleneck_features_path = 'weights/resnet50_ft_bottleneck_features_train.npy'
    val_bottleneck_features_path = 'weights/resnet50_ft_bottleneck_features_validation.npy'
    top_model_weights_path = 'weights/resnet50_ft_bottleneck_fc_model.h5'
    model_path = 'weights/resnet50_ft_model.h5'
    history_path = 'histories/resnet50_ft_history'

    defaults = {
        'img_width': 224,
        'img_height': 224,
        'batch_size': 16,
        'epochs': 30
    }

    img_width = 0
    img_height = 0

    def __init__(self, img_width=defaults['img_width'], img_height=defaults['img_height']):
        self.img_width = img_width
        self.img_height = img_height

    def train(self, train_data_dir, val_data_dir, epochs=defaults['epochs'], batch_size=defaults['batch_size']):
        self.save_bottleneck_features(train_data_dir, val_data_dir, batch_size)
        self.train_top_model(train_data_dir, val_data_dir, epochs, batch_size)
        return self.fine_tune(train_data_dir, val_data_dir, epochs, batch_size)

    def save_bottleneck_features(self, train_data_dir, val_data_dir, batch_size):
        train_generator = util.get_generator(train_data_dir, self.img_width, self.img_height, batch_size)
        val_generator = util.get_generator(val_data_dir, self.img_width, self.img_height, batch_size)

        model = self.get_base_model()
        util.save_model_plot(self.base_model_plot_path, model)

        train_bottleneck_features = model.predict_generator(
            train_generator, len(train_generator.filenames) // batch_size)
        util.save_bottleneck_features(self.train_bottleneck_features_path, train_bottleneck_features)

        val_bottleneck_features = model.predict_generator(
            val_generator, len(val_generator.filenames) // batch_size)
        util.save_bottleneck_features(self.val_bottleneck_features_path, val_bottleneck_features)

    def train_top_model(self, train_data_dir, val_data_dir, epochs, batch_size):
        train_generator = util.get_generator(train_data_dir, self.img_width, self.img_height, batch_size)
        val_generator = util.get_generator(val_data_dir, self.img_width, self.img_height, batch_size)

        train_data = util.load_bottleneck_features(self.train_bottleneck_features_path)
        val_data = util.load_bottleneck_features(self.val_bottleneck_features_path)

        num_classes = train_generator.num_classes

        train_labels = train_generator.classes
        train_labels = to_categorical(train_labels, num_classes=num_classes)
        val_labels = val_generator.classes
        val_labels = to_categorical(val_labels, num_classes=num_classes)

        model = self.get_top_model(train_data.shape[1:], num_classes)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        util.save_model_plot(self.top_model_plot_path, model)

        model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(val_data, val_labels))
        model.save_weights(self.top_model_weights_path)
        util.eval_model_loss_acc(model, val_data, val_labels, batch_size)

    def fine_tune(self, train_data_dir, val_data_dir, epochs, batch_size):
        train_generator = util.get_categorical_generator(train_data_dir, self.img_width, self.img_height, batch_size)
        val_generator = util.get_categorical_generator(val_data_dir, self.img_width, self.img_height, batch_size)

        base_model = self.get_base_model()
        layer_num = len(base_model.layers)
        for layer in base_model.layers[:int(layer_num * 0.95)]:
            layer.trainable = False
        for layer in base_model.layers[int(layer_num * 0.95):]:
            layer.trainable = True

        top_model = self.get_top_model(base_model.output_shape[1:], train_generator.num_classes)
        top_model.load_weights(self.top_model_weights_path)
        top_model.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(top_model)
        model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        util.save_model_plot(self.model_plot_path, model)

        history = model.fit_generator(train_generator,
                                      verbose=1,
                                      steps_per_epoch=train_generator.samples / batch_size,
                                      epochs=epochs,
                                      validation_data=val_generator,
                                      validation_steps=val_generator.samples / batch_size)
        model.save(self.model_path)
        util.save_history(self.history_path, history)
        return history

    def evaluate(self, data_dir, batch_size=defaults['batch_size']):
        test_generator = util.get_generator(data_dir, self.img_width, self.img_height, batch_size)

        model = self.get_trained_model()
        test_loss, test_acc = model.evaluate_generator(test_generator)

        print('Test accuracy: ', test_acc)
        print('Test loss: ', test_loss)

    def make_prediction(self, path):
        image = util.load_image(path, self.img_width, self.img_height)
        model = self.get_trained_model()
        return model.predict_classes(image)

    def get_history(self):
        return util.load_history(self.history_path)

    def get_confusion_matrix(self, directory, batch_size=defaults['batch_size']):
        generator = util.get_generator(directory, self.img_width, self.img_height, batch_size)
        model = self.get_trained_model()
        true_labels = generator.classes
        predictions = model.predict_generator(generator)
        pred_labels = np.argmax(predictions, axis=-1)
        return confusion_matrix(true_labels, pred_labels)

    @staticmethod
    def get_top_model(input_shape, num_classes):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        return model

    def get_base_model(self):
        return ResNet50(include_top=False, weights='imagenet', input_shape=(self.img_width, self.img_height, 3))

    def get_trained_model(self):
        return load_model('../' + self.model_path)

from pathlib import Path

import keras
import numpy as np
from keras.applications import ResNet50
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix


class MyResNet50:
    resnet50_model_plot_path = 'plots/resnet50_model_plot.png'
    top_model_plot_path = 'plots/resnet50_top_model_plot.png'
    train_bottleneck_features_path = 'weights/resnet50_bottleneck_features_train.npy'
    val_bottleneck_features_path = 'weights/resnet50_bottleneck_features_validation.npy'
    top_model_weights_path = 'weights/resnet50_bottleneck_fc_model.h5'

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

    def train(self, train_generator, val_generator):
        # self.save_bottleneck_features(train_generator, val_generator)
        return self.train_top_model(train_generator, val_generator)

    def make_prediction(self, image):
        base_model = self.get_base_model()
        bottleneck_prediction = base_model.predict(image)
        top_model = self.get_top_model(bottleneck_prediction.shape[1:])
        self.load_top_model_weights(top_model)
        return top_model.predict_classes(bottleneck_prediction)

    def get_confusion_matrix(self, val_generator):
        validation_data = self.load_val_bottleneck_features()
        train_labels = val_generator.classes
        top_model = self.get_top_model(validation_data.shape[1:])
        self.load_top_model_weights(top_model)
        predicted_labels = top_model.predict_classes(validation_data)
        return confusion_matrix(train_labels, predicted_labels)

    def save_bottleneck_features(self, train_generator, val_generator):
        model = self.get_base_model()
        self.plot_base_model(model)

        train_bottleneck_features = model.predict_generator(
            train_generator, len(train_generator.filenames) // self.batch_size)
        self.save_train_bottleneck_features(train_bottleneck_features)

        val_bottleneck_features = model.predict_generator(
            val_generator, len(val_generator.filenames) // self.batch_size)
        self.save_val_bottleneck_features(val_bottleneck_features)

    def train_top_model(self, train_generator, val_generator):
        train_data = self.load_train_bottleneck_features()
        validation_data = self.load_val_bottleneck_features()

        train_labels = train_generator.classes
        train_labels = to_categorical(train_labels, num_classes=self.num_classes)
        validation_labels = val_generator.classes
        validation_labels = to_categorical(validation_labels, num_classes=self.num_classes)

        model = self.get_top_model(train_data.shape[1:])
        model.compile(optimizer=RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        self.plot_top_model(model)

        # early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        # callbacks=[early_stopping]
        history = model.fit(train_data, train_labels,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(validation_data, validation_labels),
                            validation_split=0.2)
        model.save_weights(self.top_model_weights_path)
        self.eval_model_loss_acc(model, validation_data, validation_labels)
        return history

    def get_top_model(self, input_shape):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def load_top_model_weights(self, model):
        my_file = Path(self.top_model_weights_path)
        if my_file.is_file():
            model.load_weights(self.top_model_weights_path)

    def get_base_model(self):
        model = ResNet50(include_top=False, weights='imagenet', input_shape=(self.img_width, self.img_height, 3))
        return model

    def save_val_bottleneck_features(self, val_bottleneck_features):
        np.save(open(self.val_bottleneck_features_path, 'wb'), val_bottleneck_features)

    def save_train_bottleneck_features(self, bottleneck_features_train):
        np.save(open(self.train_bottleneck_features_path, 'wb'), bottleneck_features_train)

    def load_val_bottleneck_features(self):
        validation_data = np.load(open(self.val_bottleneck_features_path, 'rb'))
        return validation_data

    def load_train_bottleneck_features(self):
        train_data = np.load(open(self.train_bottleneck_features_path, 'rb'))
        return train_data

    def plot_base_model(self, model):
        print(model.summary())
        plot_model(model, to_file=self.resnet50_model_plot_path, show_shapes=True, show_layer_names=True)

    def plot_top_model(self, model):
        print(model.summary())
        plot_model(model, to_file=self.top_model_plot_path, show_shapes=True, show_layer_names=True)

    def eval_model_loss_acc(self, model, validation_data, validation_labels):
        (eval_loss, eval_accuracy) = model.evaluate(
            validation_data, validation_labels, batch_size=self.batch_size, verbose=1)
        print("[INFO] Accuracy: {:.2f}%".format(eval_accuracy * 100))
        print("[INFO] Loss: {}".format(eval_loss))

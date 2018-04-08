from pathlib import Path

import numpy as np
import scipy
from keras import applications
from keras import backend as K
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix


class MyVgg16:
    vgg16_model_plot_path = 'plots/vgg16_model_plot.png'
    top_model_plot_path = 'plots/vgg16_top_model_plot.png'
    train_bottleneck_features_path = 'weights/vgg16_bottleneck_features_train.npy'
    val_bottleneck_features_path = 'weights/vgg16_bottleneck_features_validation.npy'
    top_model_weights_path = 'weights/vgg16_bottleneck_fc_model.h5'

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
        self.save_bottleneck_features(train_generator, val_generator)
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
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
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
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def load_top_model_weights(self, model):
        my_file = Path(self.top_model_weights_path)
        if my_file.is_file():
            model.load_weights(self.top_model_weights_path)

    @staticmethod
    def get_base_model():
        model = applications.VGG16(include_top=False, weights='imagenet')
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
        plot_model(model, to_file=self.vgg16_model_plot_path, show_shapes=True, show_layer_names=True)

    def plot_top_model(self, model):
        print(model.summary())
        plot_model(model, to_file=self.top_model_plot_path, show_shapes=True, show_layer_names=True)

    def eval_model_loss_acc(self, model, validation_data, validation_labels):
        (eval_loss, eval_accuracy) = model.evaluate(
            validation_data, validation_labels, batch_size=self.batch_size, verbose=1)
        print("[INFO] Accuracy: {:.2f}%".format(eval_accuracy * 100))
        print("[INFO] Loss: {}".format(eval_loss))

    def vis_filter(self, img_width, img_height, model, layer_name, filter_index):
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
        input_img = layer_dict[layer_name].input
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([input_img], [loss, grads])
        step = 1.

        input_img_data = (np.random.random((1, img_width, img_height, 3)) - 0.5) * 20 + 128
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

        img = input_img_data[0]
        img = self.deprocess_img(img)
        scipy.misc.toimage(img, cmin=0, cmax=255).save('filters/%s_filter_%d.png' % (layer_name, filter_index))

    def visualize_filters(self, img_width, img_height):
        model = self.get_base_model()
        for i in range(64):
            self.vis_filter(img_width, img_height, model, 'block1_conv1', i)

    def deprocess_img(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

from keras.applications import ResNet50
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

from rappers import util


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

    def train(self):
        self.save_bottleneck_features()
        return self.train_top_model()

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
        util.save_model_plot(self.resnet50_model_plot_path, model)

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

    def make_prediction(self, image):
        base_model = self.get_base_model()
        bottleneck_prediction = base_model.predict(image)
        top_model = self.get_top_model(bottleneck_prediction.shape[1:])
        util.load_model_weights(self.top_model_weights_path, top_model)
        return top_model.predict_classes(bottleneck_prediction)

    def get_confusion_matrix(self, directory):
        validation_data = util.load_bottleneck_features(self.val_bottleneck_features_path)
        val_generator = util.get_generator(
            directory,
            self.img_width,
            self.img_height,
            self.batch_size)
        train_labels = val_generator.classes
        top_model = self.get_top_model(validation_data.shape[1:])
        util.load_model_weights(self.top_model_weights_path, top_model)
        predicted_labels = top_model.predict_classes(validation_data)
        return confusion_matrix(train_labels, predicted_labels)

    def get_top_model(self, input_shape):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def get_base_model(self):
        return ResNet50(include_top=False, weights='imagenet', input_shape=(self.img_width, self.img_height, 3))

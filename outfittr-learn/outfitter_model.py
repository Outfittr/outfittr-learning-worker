"""

learn.py

@author Daquaris Chadwick

The core learning class.

MAKE THIS FASTER!!!!

"""

# TensorFlow and tf.keras
import h5py
import tensorflow as tf
import math
import numpy as np
from keras.callbacks import TensorBoard
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, load_model
from keras.optimizers import SGD


def parse_labels(labels, label_count=10):

    ret_array = []
    for label in labels:
        out = np.zeros((label_count,))
        out[label] = 1
        ret_array.append(out)
    return np.array(ret_array)


class OutfitterModel:
    def __init__(self):
        # Feature Extractor - Test other feature extractors
        self.base_model = ResNet50(weights='imagenet', include_top=False)

    def get_features(self, x):
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.base_model.predict(x)

    @staticmethod
    def process_outfit_features(outfit_features):
        feature_vector_final = None
        for feature in outfit_features:
            if feature_vector_final is None:
                feature_vector_final = feature
            else:
                feature_vector_final = np.concatenate((feature_vector_final, feature), axis=None)
        return np.asarray(feature_vector_final)

    @staticmethod
    def create_multilayer_perceptron(input_layer_shape, class_count):
        inputs = Input(shape=input_layer_shape)

        exp = math.floor(math.log(input_layer_shape[0], 2.0))
        layer_size = 2 ** exp

        x = Dense(math.floor(layer_size/512), activation='relu')(inputs)
        x = Dense(math.floor(layer_size/512), activation='relu')(x)
        x = Dense(math.floor(layer_size/512), activation='relu')(x)
        x = Dense(math.floor(layer_size/512), activation='relu')(x)
        x = Dense(math.floor(layer_size/512), activation='relu')(x)

        predictions = Dense(class_count, activation='softmax')(x)
        return Model(inputs=inputs, outputs=predictions)

    @staticmethod
    def create_model_input_vector(train_input):
        dataset_vector = []
        for (outfit_features, environment) in train_input:
            feature_vector = OutfitterModel.process_outfit_features(outfit_features)
            input_vector = np.concatenate((feature_vector, [np.asarray(environment)]), axis=None)
            dataset_vector.append(input_vector)
        return np.asarray(dataset_vector)

    @staticmethod
    def train(train_data, classes=None, load_path=None, device='/device:GPU:0'):
        if classes is None:
            classes = [1, 2, 3, 4, 5]

        (train_input, train_output) = train_data

        tbcallback = TensorBoard(log_dir='src/', histogram_freq=0, write_graph=True, write_images=True)

        input_vector = OutfitterModel.create_model_input_vector(train_input)  # train_input after proper post processing

        with tf.device(device):
            try:
                model = load_model(load_path)
            except (ImportError, ValueError, TypeError) as e:
                model = OutfitterModel.create_multilayer_perceptron(input_vector[0].shape, len(classes))
                sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(loss='categorical_crossentropy',
                              optimizer=sgd,
                              metrics=['accuracy'])

            history = model.fit(input_vector, np.asarray(train_output),
                                epochs=20,
                                batch_size=input_vector.size,
                                callbacks=[tbcallback])
            model.save('my_model.h5')
            del model

            return history.history

    @staticmethod
    def test(test_data, load_path=None):
        try:
            (test_input, test_output) = test_data

            input_vector = OutfitterModel.create_model_input_vector(test_input)  # test_input after proper post processing

            model = load_model(load_path)
            test_acc, test_loss = model.evaluate(input_vector, np.asarray(test_output))

            return test_acc, test_loss
        except (ImportError, ValueError, TypeError) as e:
            print(e)
            return "Error"

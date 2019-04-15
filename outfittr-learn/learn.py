"""

learn.py

@author Daquaris Chadwick

The core learning class.

"""

# TensorFlow and tf.keras
import tensorflow as tf
import math
import numpy as np
from keras.callbacks import TensorBoard
from keras.applications.resnet50 import preprocess_input
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import SGD


class FeatureExtractor:

    def __init__(self, feature_extractor):
        self.base_model = feature_extractor  # ResNet50(weights='imagenet', include_top=False)

    def get_features(self, x):
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.base_model.predict(x)


def train(train_data, load_path, architecture=None, device='/device:GPU:0'):
    (train_input, train_output) = train_data

    tbcallback = TensorBoard(log_dir='src/', histogram_freq=0, write_graph=True, write_images=True)

    with tf.device(device):
        try:
            model = load_model(load_path)
        except (ImportError, ValueError, OSError) as e:
            model = architecture(train_input[0].shape, 5)
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

        history = model.fit(np.asarray(train_input), np.asarray(train_output),
                            epochs=20,
                            batch_size=32,
                            callbacks=[tbcallback])
        model.save(load_path)
        del model

        return history.history


def test(self, test_data, load_path):
    try:
        (test_input, test_output) = test_data

        model = load_model(load_path)
        test_acc, test_loss = model.evaluate(test_input, np.asarray(test_output))

        return test_acc, test_loss
    except (ImportError, ValueError, TypeError) as e:
        print(e)
        return "Error"


def process_outfit_features(outfit_features):
    feature_vector_final = None
    for feature in outfit_features:
        if feature_vector_final is None:
            feature_vector_final = feature
        else:
            feature_vector_final = np.concatenate((feature_vector_final, feature), axis=None)
    return np.asarray(feature_vector_final)


def create_model_input_vector(train_input):
    dataset_vector = []
    for (outfit_features, environment) in train_input:
        feature_vector = process_outfit_features(outfit_features)
        input_vector = np.concatenate((feature_vector, [np.asarray(environment)]), axis=None)
        dataset_vector.append(input_vector)
    return np.asarray(dataset_vector)


def parse_labels(labels, label_count=10):

    ret_array = []
    for label in labels:
        out = np.zeros((label_count,))
        out[label] = 1
        ret_array.append(out)
    return np.array(ret_array)


def create_multilayer_perceptron(input_layer_shape, class_count):
    """Create the multilayer perceptron to be used in the train method"""

    inputs = Input(shape=input_layer_shape)

    exp = math.floor(math.log(input_layer_shape[0], 2.0))
    layer_size = 2 ** exp

    x = Dense(math.floor(layer_size/512), activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(math.floor(layer_size/512), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(math.floor(layer_size/512), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(math.floor(layer_size/512), activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(class_count, activation='softmax')(x)
    return Model(inputs=inputs, outputs=predictions)
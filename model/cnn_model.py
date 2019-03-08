from keras.layers.convolutional import ( 
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
 )
from keras.layers.core import ( 
    Activation,
    Dense,
    Dropout,
    Flatten
)

from keras import layers
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

import numpy as np


class LaeCNN():

    def __init__(self, input_shape, lr=0.001):
        self.input_shape = input_shape
        self.model = self._build_cnn_model()
        optimizer = SGD(lr=lr)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def _build_cnn_model(self):
        model = Sequential()

        model.add(Conv2D(10, (3,3), input_shape=self.input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model
    
    def __build_residual_block(self, x, filters, kernel_size, strides=(1,1)):

        data = x
        x = layers.Conv2d(filters, kernel_size, strides=strides, padding='same')(input)
        x = layers.normalization.BatchNormalization(activation='relu')(x)
        data = Conv2D(filters=int(x.shape[3]), kernel_size=(1,1), strides=strides, padding="same")(data)
    
    def _build_vgg_model(self):
        model = Sequential()

        model.add(Conv2D(64, (3,3), input_shape=self.input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

        model.add(Flatten())
        model.add(Dense(4069))
        model.add(Activation('relu'))
        model.add(Dense(4069))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model

    
    def train(self, epoch, batch_size, train_data, train_label):
        train_data_np = np.array(train_data)
        train_data_np = train_data_np.reshape(train_data_np.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2])
        train_label_np = np.array(train_label)
        train_label_np = to_categorical(train_label_np)
        return self.model.fit(
            train_data_np, train_label_np, epochs=epoch, batch_size=batch_size, verbose=0
        )

    def test(self, test_data, test_label):
        test_data_np = np.array(test_data)
        test_data_np = test_data_np.reshape(test_data_np.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2])
        test_label_np = np.array(test_label)
        test_label_np = to_categorical(test_label)
        #score = self.model.evaluate(test_data_np, test_label_np, verbose=0)
        prob = self.model.predict(test_data_np)
        pred = self.model.predict_classes(test_data_np)
        return None, pred, prob
    
    def save_model(self, output_path):
        json_model = self.model.to_json()
        with open(output_path, 'w') as f:
            f.write(json_model)

    def save_weights(self, output_path):
        self.model.save_weights(output_path)
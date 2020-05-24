# @author: Khushal Brahmbhatt

from keras.layers import Lambda, Conv2D, Flatten, Dense, ELU, Dropout
from keras.models import Sequential


class NvidiaDave2:
    @staticmethod
    def build(rows, cols, channels):
        model = Sequential()

        # Input normalization layer
        model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(rows, cols, channels)))

        # 5x5 convolutional layers with stride of 2x2
        model.add((Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='elu')))
        model.add((Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='elu')))
        model.add((Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='elu')))

        # 3x3 convolutional layers with stride of 1x1
        model.add((Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu')))
        model.add((Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu')))

        # Flatten before passing to the fully connected layers
        model.add(Flatten())

        # Three fully connected layers
        model.add(Dense(100))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(50))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(10))
        model.add(Dropout(.5))
        model.add(ELU())

        # Output layer with tanh activation
        model.add(Dense(1, activation='tanh'))

        return model

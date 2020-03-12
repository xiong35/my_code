import keras
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.regularizers import l2
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


def train(model):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_test /= 255
    train_datagan = ImageDataGenerator(rescale=1./255, )
    # test_datagen = ImageDataGenerator(rescale=1./255)
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir='my_log'
        )
    ]

    hist = model.fit_generator(train_datagan.flow(x_train, y_train, batch_size=32),
                               steps_per_epoch=25,
                               epochs=1, validation_data=(x_test, y_test), shuffle=True,
                               callbacks=callbacks)

    history = hist.history
    acc = history['acc']
    loss = history['loss']
    epochs = range(1, len(acc)+1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.legend()
    plt.savefig('densenet_acc.png')
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.legend()
    plt.savefig('densenet_loss.png')


def conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4
        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(ip, nb_filter, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * 0.5), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


nb_classes = 10
depth = 30
nb_dense_block = 2

growth_rate = 16
nb_filter = 32
nb_layers_per_block = [6, 6]
bottleneck = True

dropout_rate = 0.2
weight_decay = 1e-4
subsample_initial_block = True
activation = 'softmax'
input_shape = (32, 32, 3)

inputs = Input(shape=input_shape)
x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same',
           strides=1,  kernel_regularizer=l2(weight_decay))(inputs)

x_list = [x]

for i in range(3):
    cb = conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
    x_list.append(cb)
    x = concatenate([x, cb], axis=-1)
    nb_filter += growth_rate

x = transition_block(
    x, nb_filter,  weight_decay=weight_decay)

nb_filter = int(nb_filter * 0.7)

for i in range(3):
    cb = conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
    x_list.append(cb)
    x = concatenate([x, cb], axis=-1)
    nb_filter += growth_rate

x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(nb_classes, activation=activation)(x)


model = Model(inputs, x)

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])

train(model)

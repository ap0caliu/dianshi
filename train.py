#!/usr/bin/python3.6
#-*- coding:utf-8 -*-


import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from MyGenerator import myGenerator
import os
import matplotlib.pyplot as plt
import numpy as np


# global vars
workspace = os.getcwd()
classnames = ['OCEAN', 'MOUNTAIN', 'LAKE', 'FARMLAND', 'DESERT', 'CITY']
n_class = len(classnames)
# EPOCHES = 20
EPOCHES = 20
PATCH_SIZE = (256, 256)
BATCH_SIZE = 10
LR = 0.001
ratio = 0.7

if not os.path.exists('./logs'): os.mkdir('./logs')

def Network():
    input = Input(shape=(PATCH_SIZE[0], PATCH_SIZE[1], 3))
    conv_1 = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu')(input)
    maxpool_1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv_1)
    flatten = Flatten()(maxpool_1)
    dense_1 = Dense(40, activation = 'relu')(flatten)
    output = Dense(n_class, activation = 'softmax')(dense_1)
    model = Model(inputs = input, outputs = output)
    optim = keras.optimizers.SGD(lr=LR, momentum=0.9, decay=LR/EPOCHES, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
    model.summary()
    return model

def trainProcess():
    callback = [keras.callbacks.ModelCheckpoint('./logs/{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc',
                                               verbose=2, save_best_only=False, save_weights_only=True, mode='max')]
    model = Network()
    mygenerator = myGenerator(ratio, PATCH_SIZE, BATCH_SIZE, classnames, workspace)
    history = model.fit_generator(mygenerator.generator(trainflag=True), steps_per_epoch=mygenerator.train_batch, epochs=EPOCHES, verbose=1, callbacks=callback, validation_data=mygenerator.generator(trainflag=False), validation_steps=mygenerator.valid_batch)
    params = history.history
    x = np.arange(1, EPOCHES+1)
    acc = params["acc"]
    loss = params["loss"]
    val_acc = params["val_acc"]
    val_loss = params["val_loss"]
    # print(x, acc, loss, val_acc, val_loss)
    plt.figure()
    plt.plot(x, acc, 'r', label='acc')
    plt.plot(x, loss, 'g', label='loss')
    plt.plot(x, val_acc, 'b', label='val_acc')
    plt.plot(x, val_loss, 'y', label='val_loss')
    plt.legend()
    plt.savefig('./image.png')
    plt.show()




if __name__ == "__main__":
    trainProcess()
    K.clear_session()

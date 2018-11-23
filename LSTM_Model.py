#!/usr/bin/env python
# -*- coding:utf-8 -*-
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers import Input, Dense, TimeDistributed, Lambda
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.regularizers import l2,l1
from keras.layers import Dense, Activation, Dropout, LSTM

def lstm_simple(
        n_classes,
        neurons,
        feat_dim,
        max_len,
        dropout=0.2,
        activation="softmax"):
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    input = Input(shape=(max_len,feat_dim))
    lstm = LSTM(neurons)(input)
    dr = Dropout(dropout)(lstm)
    flatten = Flatten()(dr)
    # model = Dense(n_classes,activation=activation)(flatten)
    dense = Dense(units=n_classes,
                  activation=activation,
                  kernel_initializer="he_normal")(flatten)

    model = Model(inputs=input, outputs=dense)
    return model






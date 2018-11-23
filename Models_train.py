#!/usr/bin/env python
# -*- coding:utf-8 -*-

import  os
import lmdb
import numpy as np
import TCN_Model
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2,l1
import pdb
import numpy as np
import lmdb
import threading
import os
from keras.callbacks import ReduceLROnPlateau
from LSTM_Model import *

os.environ['CUDA_VISIBLE_DEVICES']='0'

np.random.seed(seed=1234)

data_root = r'/home/xsh/my_file/project/Graduate_work/010_data'

lmdb_file_train_x = os.path.join(data_root,'Xtrain_lmdb')
lmdb_file_train_y = os.path.join(data_root,'Ytrain_lmdb')
lmdb_file_test_x = os.path.join(data_root,'Xtest_lmdb')
lmdb_file_test_y = os.path.join(data_root,'Ytest_lmdb')


# 0:TCN_simple, 1:TCN_plain, 2:TCN_resnet, 3:TCN_simple_resnet,4:lstm
model_choice = 2
loss = 'categorical_crossentropy'
lr = 0.01
momentum = 0.9
activation = "relu"
optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)
dropout = 0.5
reg = l1(1.e-4)
out_dir_names = ['TCN_raw_resnet_L10','lstm']
out_dir_name = out_dir_names[1]
augment = 0




## TRAINING PARAMS
batch_size = 32
epochs = 200
verbose = 1
shuffle = 1
feat_dim =75
n_classes = 2
factor_denom = 2
neurons = 256

max_len = 169 #一个步态数据的最大帧率长度

samples_per_epoch = 411#训练样本数量
samples_per_validation = 173 #测试样本数量

if augment:
  num_training_samples = samples_per_epoch*4
else:
  num_training_samples = samples_per_epoch



def gait_train_datagen(augmentation=1):
    lmdb_env_x = lmdb.open(lmdb_file_train_x)
    lmdb_txn_x = lmdb_env_x.begin()
    lmdb_cursor_x = lmdb_txn_x.cursor()

    lmdb_env_y = lmdb.open(lmdb_file_train_y)
    lmdb_txn_y = lmdb_env_y.begin()
    lmdb_cursor_y = lmdb_txn_y.cursor()

    X = np.zeros((batch_size, max_len, feat_dim))
    Y = np.zeros((batch_size, n_classes))
    batch_count = 0
    while True:
        indices = list(range(0, samples_per_epoch))
        np.random.shuffle(indices)

        for index in indices:
            # pre_value = lmdb_cursor_x.get('{:0>8d}'.format(index).encode())
            # print("pre_value:",pre_value)
            # print("pre_value len: ",len(pre_value))

            value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
            # print("value",value)
            # print("value shape ",value.shape)
            # print(sum(value > 0 ))

            # print("label_pre_len:",len(label_pre))
            label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()), dtype=np.float32)
            # print("label:\n",label)

            ## THIS IS MEAN SUBTRACTION
            x = value.reshape((max_len, feat_dim))
            nonzeros = np.where(np.array([np.sum(x[i]) > 0 for i in range(0, x.shape[0])]) == False)[0]  # 找出<=0的位置
            # value.reshape((max_len,feat_dim))
            if len(nonzeros) == 0:
                last_time = 0
            else:
                last_time = nonzeros[0]
            x.setflags(write=1)
            # x[:last_time] = x[:last_time] - train_x_mean  # ？？？？因为有些帧数可能比较短达不到max_len,所以后面为0

            ## ORIGINAL
            # X[batch_count] =  x.reshape(max_len,feat_dim,1)#???? #Todo
            X[batch_count] = x.reshape(max_len, feat_dim)
            # print("=========")
            # print(label.shape)
            # print(label)
            Y[batch_count] = label
            batch_count += 1

            if augmentation:
                ## TEMPORAL SHIFT
                shift_range = np.random.randint(10) + 1
                shift_augment_x = np.zeros(x.shape)
                end_point = min(shift_range + last_time, max_len)
                shift_augment_x[shift_range:end_point] = x[:last_time]
                X[batch_count] = shift_augment_x
                Y[batch_count] = label
                batch_count += 1

                if batch_count == batch_size:
                    ret_x = X
                    ret_y = Y
                    X = np.zeros((batch_size, max_len, feat_dim))
                    Y = np.zeros((batch_size, n_classes))
                    batch_count = 0
                    yield (ret_x, ret_y)

                ## TEMPORAL STRETCH
                if last_time > 0:
                    factor = np.random.random() / factor_denom + 1
                    new_length = min(int(last_time * factor), max_len)
                    stretch_augment_x = np.zeros(x.shape)
                    stretched = np.resize(x[:last_time], (new_length, feat_dim))
                    # stretched = cv2.resize(x[:last_time],(feat_dim,new_length),interpolation=cv2.INTER_LINEAR)
                    stretch_augment_x[:new_length] = stretched
                    X[batch_count] = stretch_augment_x
                    Y[batch_count] = label
                    batch_count += 1

                if batch_count == batch_size:
                    ret_x = X
                    ret_y = Y
                    X = np.zeros((batch_size, max_len, feat_dim))
                    Y = np.zeros((batch_size, n_classes))
                    batch_count = 0
                    yield (ret_x, ret_y)

                ## TEMPORAL SHRINK
                if last_time > 0:
                    factor = 1 - np.random.random() / factor_denom
                    new_length = max(int(last_time * factor), 1)
                    shrink_augment_x = np.zeros(x.shape)
                    shrinked = np.resize(x[:last_time], (new_length, feat_dim))
                    # shrinked = cv2.resize(x[:last_time],(feat_dim,new_length),interpolation=cv2.INTER_LINEAR)
                    shrink_augment_x[:new_length] = shrinked
                    X[batch_count] = shrink_augment_x
                    Y[batch_count] = label
                    batch_count += 1

            if batch_count == batch_size:
                ret_x = X
                ret_y = Y
                X = np.zeros((batch_size, max_len, feat_dim))
                Y = np.zeros((batch_size, n_classes))
                batch_count = 0
                yield (ret_x, ret_y)

def gait_test_datagen():
    lmdb_env_x = lmdb.open(lmdb_file_test_x)
    lmdb_txn_x = lmdb_env_x.begin()
    lmdb_cursor_x = lmdb_txn_x.cursor()

    lmdb_env_y = lmdb.open(lmdb_file_test_y)
    lmdb_txn_y = lmdb_env_y.begin()
    lmdb_cursor_y = lmdb_txn_y.cursor()

    X = np.zeros((batch_size, max_len, feat_dim))
    Y = np.zeros((batch_size, n_classes))

    while True:
        indices = list(range(0, samples_per_validation))
        np.random.shuffle(indices)
        batch_count = 0
        for index in indices:
            value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()))
            label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index).encode()), dtype=np.float32)

            ## THIS IS MEAN SUBTRACTION
            x = value.reshape((max_len, feat_dim))
            nonzeros = np.where(np.array([np.sum(x[i]) > 0 for i in range(0, x.shape[0])]) == False)[0]
            # value.reshape((max_len,feat_dim))

            if len(nonzeros) == 0:
                last_time = 0
            else:
                last_time = nonzeros[0]
            x.setflags(write=1)
            # x[:last_time] = x[:last_time] - train_x_mean

            X[batch_count] = x.reshape(max_len, feat_dim)
            Y[batch_count] = label

            batch_count += 1

            if batch_count == batch_size:
                ret_x = X
                ret_y = Y
                X = np.zeros((batch_size, max_len, feat_dim))
                Y = np.zeros((batch_size, n_classes))
                batch_count = 0
                yield (ret_x, ret_y)


def train():
    model_TCN_simple = TCN_Model.TCN_simple(
        n_classes,
        feat_dim,
        max_len,
        gap=1,
        dropout=dropout,
        kernel_regularizer=l2(1.e-4),
        activation=activation)
    model_TCN_plain = TCN_Model.TCN_plain(
        n_classes,
        feat_dim,
        max_len,
        gap=1,
        dropout=dropout,
        kernel_regularizer=reg,
        activation=activation)
    model_TCN_resnet = TCN_Model.TCN_resnet(
        n_classes,
        feat_dim,
        max_len,
        gap=1,
        dropout=dropout,
        kernel_regularizer=reg,
        activation=activation)
    model_TCN_simple_resnet = TCN_Model.TCN_simple_resnet(
        n_classes,
        feat_dim,
        max_len,
        gap=1,
        dropout=dropout,
        kernel_regularizer=reg,
        activation=activation)

    # model_lstm = lstm_simple(
    #     n_classes,
    #     neurons,
    #     feat_dim,
    #     max_len,
    #     dropout=0.2,
    #     activation="softmax")

    # models = [model_TCN_simple, model_TCN_plain, model_TCN_resnet, model_TCN_simple_resnet, model_lstm]
    models = [model_TCN_simple, model_TCN_plain, model_TCN_resnet, model_TCN_simple_resnet]
    model = models[model_choice]

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    if not os.path.exists('weights/' + out_dir_name):
        os.makedirs('weights/' + out_dir_name)

    weight_path = 'weights/' + out_dir_name + '/{epoch:03d}_{val_acc:0.3f}.hdf5'

    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=10,
                                  verbose=1,
                                  mode='auto',
                                  cooldown=3,
                                  min_lr=0.0001)

    callbacks_list = [checkpoint, reduce_lr]

    model.fit_generator(gait_train_datagen(augment),
                        steps_per_epoch=num_training_samples / batch_size + 1,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=gait_test_datagen(),
                        validation_steps=samples_per_validation / batch_size + 1,
                        workers=1,
                        initial_epoch=0
                        )


#todo 计算数据库的平均值



if __name__ == "__main__":
  train()
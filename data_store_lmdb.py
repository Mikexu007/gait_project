#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import pandas as pd
import numpy as np
from keras.utils import np_utils
import lmdb


n_classes = 2
map_size = int(1e12)
batch_size = 128


def read_one_skeleton_file(root_dir):
    sds_hospital = pd.read_table(root_dir, header=None, delimiter=' ')
    np_arr = np.array(sds_hospital)
    return np_arr


def generate_data(collector_dir,label_dir,data_out_dir,day_or_view):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    training_day = []
    training_view = [1,2,4,5]

    files = os.listdir(collector_dir)

    max_len = 0

    id_degree = pd.read_table(label_dir, header=None, delimiter=' ')

    for file in files:
        name_splits = file.split("_")
        id = name_splits[0]
        view = name_splits[1][-1]
        day = name_splits[2].split('-')[1]

        if int(view) == 7 :
            continue
        if int(view) == 8 :
            continue

        #直接找到对应的label
        target_row = id_degree[id_degree[0].isin([id])]
        if list(target_row[1]) != []:
            temp = list(target_row[1])
            label = temp[0]
            # label = int(target_row[1])

            file_dir = os.path.join(collector_dir, file)
            frames = read_one_skeleton_file(file_dir)
            len_frames = len(frames)

            if day_or_view == "view":
                if int(view) in training_view:
                    X_train.append(frames)
                    Y_train.append(label)
                else:
                    X_test.append(frames)
                    Y_test.append(label)

            if day_or_view == "day":
                if int(day) in training_day:
                    X_train.append(frames)
                    Y_train.append(label)
                else:
                    X_test.append(frames)
                    Y_test.append(label)

            max_len = max(max_len,len_frames)

    lmdb_file_x = os.path.join(data_out_dir, 'Xtrain_lmdb')
    lmdb_file_y = os.path.join(data_out_dir, 'Ytrain_lmdb')

    lmdb_env_x = lmdb.open(lmdb_file_x, map_size=map_size)
    # lmdb_env_x = lmdb.open(lmdb_file_x)
    lmdb_env_y = lmdb.open(lmdb_file_y, map_size=map_size)
    # lmdb_env_y = lmdb.open(lmdb_file_y)
    lmdb_txn_x = lmdb_env_x.begin(write=True)
    lmdb_txn_y = lmdb_env_y.begin(write=True)

    item_id = -1
    for i in range(0, len(X_train)):
        item_id += 1
        keystr = '{:0>8d}'.format(item_id)
        X = np.zeros((max_len, 75))
        num_rows = X_train[i].shape[0]
        X[0:num_rows] = X_train[i]
        Y = np_utils.to_categorical(Y_train[i], n_classes)
        # print("Y:", Y)
        # print("Y len:",len(Y))
        # print("X:",X)
        # print("X.tobytes:",X.tobytes())

        lmdb_txn_x.put(keystr.encode(), X.tobytes())
        lmdb_txn_y.put(keystr.encode(), Y.tobytes())
        # print("Y tobytes", len(Y.tobytes()))
        # print(Y.tobytes())

        if (item_id + 1) % batch_size == 0:
            lmdb_txn_x.commit()
            lmdb_txn_x = lmdb_env_x.begin(write=True)
            lmdb_txn_y.commit()
            lmdb_txn_y = lmdb_env_y.begin(write=True)
            print((item_id + 1))

    if (item_id + 1) % batch_size != 0:
        lmdb_txn_x.commit()
        lmdb_txn_y.commit()
        print('last batch')
        print((item_id + 1))

    print ("WROTE TRAINING")


    lmdb_file_x = os.path.join(data_out_dir, 'Xtest_lmdb')
    lmdb_file_y = os.path.join(data_out_dir, 'Ytest_lmdb')

    lmdb_env_x = lmdb.open(lmdb_file_x, map_size=map_size)
    # lmdb_env_x = lmdb.open(lmdb_file_x)
    lmdb_env_y = lmdb.open(lmdb_file_y, map_size=map_size)
    # lmdb_env_y = lmdb.open(lmdb_file_y)
    lmdb_txn_x = lmdb_env_x.begin(write=True)
    lmdb_txn_y = lmdb_env_y.begin(write=True)

    item_id = -1
    for i in range(0, len(X_test)):
        item_id += 1
        keystr = '{:0>8d}'.format(item_id)

        X = np.zeros((max_len, 75))
        num_rows = X_test[i].shape[0]
        X[0:num_rows] = X_test[i]
        Y = np_utils.to_categorical(Y_test[i], n_classes)

        lmdb_txn_x.put(keystr.encode(), X.tobytes())
        lmdb_txn_y.put(keystr.encode(), Y.tobytes())

        # write batch
        if (item_id + 1) % batch_size == 0:
            lmdb_txn_x.commit()
            lmdb_txn_x = lmdb_env_x.begin(write=True)
            lmdb_txn_y.commit()
            lmdb_txn_y = lmdb_env_y.begin(write=True)
            print((item_id + 1))

    if (item_id + 1) % batch_size != 0:
        lmdb_txn_x.commit()
        lmdb_txn_y.commit()
        print('last batch')
        print((item_id + 1))

    print("WROTE TESTING")
    print(("TRAINING SAMPLES: ", len(X_train), "TESTING SAMPLES:", len(X_test)))
    print("max_len %d"%max_len)
    return max_len,len(X_train),len(X_test)

collector_dir = '/home/xsh/my_file/project/Graduate_work/010_data/training_test_data'
label_dir = r'/home/xsh/my_file/project/Graduate_work/010_data/gait/all_sds_shuffle.csv'
data_out_dir = r'/home/xsh/my_file/project/Graduate_work/010_data'
day_or_view = 'view'
max_len, training_samples, test_samples = generate_data(collector_dir,label_dir,data_out_dir,day_or_view)


result_dir = os.path.join(data_out_dir,'data_store_lmdb_result.txt')
f = open(result_dir,mode='w')
header = 'max_len, training_samples, test_samples '
result = str(max_len) + ','+ str(training_samples)+ ',' + str(test_samples)
f.write(header)
f.write('\n')
f.write(result)
f.close()
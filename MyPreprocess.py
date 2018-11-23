#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2016 Inwoong Lee All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the MSR Action3D text file format."""
import tensorflow as tf
import numpy as np
import os
import random
import csv
import pandas as pd

def read_all_skeleton(DATA_PATH, flag, system):

    dict = {}
    index = []

    count = 0
    rootdir = DATA_PATH
    list = os.listdir(rootdir)
    # print(list)
    for i in list:
        path = os.path.join(rootdir, i) # e.g. .../p122
        # print("path:",str(path).split('\\')[-1][1:])
        if os.path.isdir(path):
            sublist = os.listdir(path)
            for j in sublist:
                file = open(os.path.join(path,j))
                csv_reader = csv.reader(file,delimiter=" ")
                temp_sklt = []
                for row in csv_reader:
                    temp_sklt.append(row)# 输出list
                    # print(row)
                count +=1

                first_letter = str(i)[0]
                if first_letter == 'p':
                    num = int(str(i)[1:])
                else:
                    num = int(str(i).split('_')[0])

                if num in index:
                    dict[num].append(temp_sklt)
                else:
                    dict[num] = []
                    dict[num].append(temp_sklt)
                    index.append(num)
                file.close()
    print("total file:",count)
    # print(dict)
    return dict





def read_excel_p_pp(DATA_PATH, file_name, class_size, flag):
    excel = pd.read_excel(os.path.join(DATA_PATH,file_name))
    # print(excel)
    num_level = excel.loc[:,['number','d_level']]

    if class_size == 2:
        num_level = num_level.replace({"无":0,"轻": 1, "中": 1, "重": 1})
        if flag == 'd':
            output = num_level[num_level['d_level'] == 1]
        else:
            output = num_level[num_level['d_level'] == 0]

    else:
        num_level = num_level.replace({"无": 0, "轻": 1, "中": 2, "重": 3})
        # a=num_level.iloc[1,1] 变换为int了
        if flag == 'd':
            output = num_level[np.logical_or(np.array(num_level['d_level']) == 1,
                                         np.array(num_level['d_level']) == 2, np.array(num_level['d_level']) == 3)]
        else:
            output = num_level[num_level['d_level'] == 0]


    # print(output)
    return output
    # print(outfile)

# def read_excel_h(DATA_PATH,file_name,class_size,flag):
#     excel = pd.read_excel(os.path.join(DATA_PATH, file_name))
#     # print(excel)
#     num_level = excel.loc[:, ['量表编号', '结果']]
#     num_level.rename(columns={'量表编号':'number','结果':'d_level'},inplace=True)
#     # print(num_level)
#     if class_size == 2:
#         num_level = num_level.replace({"无抑郁症状":0,"轻度抑郁": 1, "中度抑郁": 1, "重度抑郁": 1})
#         if flag == 'd':
#             output = num_level[num_level['d_level'] == 1]
#         else:
#             output = num_level[num_level['d_level'] == 0]
#
#     else:
#         num_level = num_level.replace({"无抑郁症状": 0, "轻度抑郁": 1, "中度抑郁": 2, "重度抑郁": 3})
#         # a=num_level.iloc[1,1] 变换为int了
#         if flag == 'd':
#             output = num_level[np.logical_or(np.array(num_level['d_level']) == 1,
#                                          np.array(num_level['d_level']) == 2, np.array(num_level['d_level']) == 3)]
#         else:
#             output = num_level[num_level['d_level'] == 0]
#
#     # print()
#     return output


#输出对应的数据和标签
def response(data_dir,label_dataframe):

    data = []
    labels = []

    label_number = label_dataframe['number']
    # print(label_number.tolist())
    label_number = label_number.tolist()

    label_d_level = label_dataframe['d_level']
    label_d_level = label_d_level.tolist()


    for i,num in enumerate(label_number):
        # print(i,":",num)
        if num in data_dir.keys():
            # print("good")
            episode_nums =len(data_dir[num])# 某个病人的总镜头数
            for episode_index in range(episode_nums):
                bone_frames = data_dir[num][episode_index]

                data.append(bone_frames)
                ###To do
                temp = []
                temp.append(label_d_level[i])
                # print(temp)
                labels.append(temp)

    # data = np.array(data)
    # labels = transform(np.array(labels),2)
    # print(data.shape)
    # print(len(data[1][0]))
    # print(labels)
    # print(labels.shape)
    return data,labels

#将label装换成 0101
def transform_label(original_label,class_size):

    new_label = np.zeros([len(original_label), class_size])

    for batch_step in range(len(original_label)):
        # print ("original_label[batch_step]",original_label[batch_step])
        new_label[batch_step][original_label[batch_step]] = 1

    return new_label

# 用于lstm需要等长帧数据，传入MAX_LENGTH进行padding
def tranform_data(source, MAX_LENGTH):
    new_data = np.zeros([len(source),MAX_LENGTH,75])

    for episode_index in range(len(source)):
        # x = pd.DataFrame(source[episode_index])
        #
        # x = x.drop([0,1], 1)
        # x = np.array(x)
        # x = x.tolist()
        # print(len(x[0]))
        # if episode_index == 0:
        #     break
        x = source[episode_index]
        new_data[episode_index][MAX_LENGTH - len(source[episode_index]):MAX_LENGTH] = x
        for time_step in range(MAX_LENGTH):
            if np.sum(new_data[episode_index][time_step]) != 0:
                for ttime_step in range(time_step):
                    new_data[episode_index][ttime_step] = new_data[episode_index][time_step]
                break
            else:
                pass
    return new_data

def countMaxLength(data):
    MAX_LENGTH = 0
    for batchNo in range(len(data)):
        if len(data[batchNo]) > MAX_LENGTH:
            MAX_LENGTH = len(data[batchNo])
        else:
            pass

    return MAX_LENGTH


#划分训练集和测试集
def data_preprocess(train_path_d,train_path_n,test_path_d,test_path_n,system_version,class_size):
    ##读txt需要分开两个文件夹，读excel需要分开
    train_data_dit_d = read_all_skeleton(train_path_d, "d", system_version)
    train_data_dit_n = read_all_skeleton(train_path_n, "n", system_version)
    test_data_dit_d = read_all_skeleton(test_path_d, "d", system_version)
    test_data_dit_n = read_all_skeleton(test_path_n, "n", system_version)

    label_df_d = read_excel_p_pp(train_path_d, "depression.xlsx", class_size, 'd')
    label_df_n = read_excel_p_pp(train_path_n, "normal.xlsx", class_size, 'n')

    # test_label_dataframe = MyPreprocess.read_excel_two(test_path)

    data_train_d, label_train_d = response(train_data_dit_d, label_df_d)
    data_train_n, label_train_n = response(train_data_dit_n, label_df_n)

    data_test_d, label_test_d = response(test_data_dit_d,label_df_d)
    data_test_n, label_test_n = response(test_data_dit_n,label_df_n)

    max_length0 = countMaxLength(data_train_d)
    max_length1 = countMaxLength(data_train_n)
    max_length2 = countMaxLength(data_test_d)
    max_length3 = countMaxLength(data_test_n)

    MAX_LENGTH = max(max_length0,max_length1,max_length2,max_length3)


    # TODO=======================
    train_d= tranform_data(data_train_d,MAX_LENGTH)  # 去掉每一帧前两个数字然后转成np.array
    train_n = tranform_data(data_train_n, MAX_LENGTH)

    test_d = tranform_data(data_test_d,MAX_LENGTH)
    test_n = tranform_data(data_test_n,MAX_LENGTH)


    train_label_d = transform_label(label_train_d, class_size)  # 变成label
    train_label_n = transform_label(label_train_n, class_size)

    test_label_d = transform_label(label_test_d, class_size)  # 变成label
    test_label_n = transform_label(label_test_n, class_size)

    temp_train_data = np.concatenate([train_d, train_n], 0)
    temp_test_data = np.concatenate([test_d, test_n], 0)

    temp_train_label = np.concatenate([train_label_d, train_label_n], 0)
    temp_test_label = np.concatenate([test_label_d, test_label_n], 0)

    # print("temp_train_data:",temp_train_data.shape)
    # print("temp_test_data:",temp_test_data.shape)
    # print("temp_train_label:",temp_train_label.shape)
    # print("temp_test_label:",temp_test_label.shape)

#####读数据进行shuffle
    temp0 = temp_train_label.shape[0]
    permutation_train = np.random.permutation(temp0)

    shuffled_train_data = temp_train_data[permutation_train,:,:]
    shuffled_train_label = temp_train_label[permutation_train,:]

    temp1 = temp_test_label.shape[0]
    permutation_test = np.random.permutation(temp1)

    shuffle_test_data = temp_test_data[permutation_test,:,:]
    shuffle_test_label = temp_test_label[permutation_test,:]
    return shuffled_train_data,shuffled_train_label,shuffle_test_data,shuffle_test_label,MAX_LENGTH

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path,i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def check(myFolderPath):
    if os.path.exists(myFolderPath):
        del_file(myFolderPath)
        os.rmdir(myFolderPath)
    else:
        os.makedirs(myFolderPath)


def separater_xyz(input):
    x_index = []
    y_index = []
    z_index = []
    for i in range(len(input[0,0])):
        if i%3 == 0:
            x_index.append(i)
        elif i%3 == 1:
            y_index.append(i)
        else:
            z_index.append(i)

    x_index = np.array(x_index)
    y_index = np.array(y_index)
    z_index = np.array(z_index)

    # print("x_index",x_index)
    x = input[:,:,x_index]
    y = input[:,:,y_index]
    z = input[:,:,z_index]
    return x,y,z
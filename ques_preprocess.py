#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import pandas as pd
import numpy as np


def read_sds_phq_output(file_dir, out_dir, class_size):
    content = pd.read_table(file_dir,header=None,delimiter='\t')
    number_sds_phq = content.iloc[:,[0,2,8]]
    # print(number_sds)
    # print(number_sds.shape)
    new_sds_phq = number_sds_phq.drop_duplicates(0,keep='last',inplace=False)

    indexs = []
    index = 0
    for i in new_sds_phq.iloc[:,0]:
        number = str(i)
        if len(number) == 12:
            indexs.append(index)
        index += 1

    new_sds_phq = new_sds_phq.iloc[indexs,:]
    sds = new_sds_phq.iloc[:,1]
    phq = new_sds_phq.iloc[:,2]
    # print(sds)
    # print(phq)
    if class_size == 2:
        sds = sds.replace({"无":0,"轻": 1, "中": 1, "重": 1})
        # print(sds)
        new_sds_phq.iloc[:, 1] = sds
        phq = phq.replace({"无":0,"轻": 1, "中": 1, "中重": 1,"重": 1})
        new_sds_phq.iloc[:, 2] = phq

    new_sds_phq.to_csv(out_dir, index=False, header=None, sep=' ')

    # return new_sds_phq



def separate_sds_phq(root_dir,out_dir1,out_dir2):
    content = pd.read_table(root_dir, header=None, delimiter=' ')
    sds = content.iloc[:,0:2]
    phq = content.iloc[:,[0,2]]
    sds.to_csv(out_dir1, index=False, header=None, sep=' ')
    phq.to_csv(out_dir2, index=False, header=None, sep=' ')


def read_excel_p_output(file_name, class_size, flag, out_dir):
    excel = pd.read_excel(file_name)
    # print(excel)
    num_level = excel.loc[:,['number','d_level']]

    temp_shape = num_level.iloc[:,1].shape
    pd_label = np.ones(temp_shape)

    if class_size == 2:
        # num_level = num_level.replace({"无":0,"轻": 1, "中": 1, "重": 1})
        num_level.iloc[:,1] = pd_label
        id_with_p = num_level.iloc[:,0].apply(lambda x:"p"+str(x))
        num_level.iloc[:,0] = id_with_p
        output = num_level
    else:
        num_level = num_level.replace({"无": 0, "轻": 1, "中": 2, "重": 3})
        # a=num_level.iloc[1,1] 变换为int了
        if flag == 'd':
            output = num_level[np.logical_or(np.array(num_level['d_level']) == 1,
                                         np.array(num_level['d_level']) == 2, np.array(num_level['d_level']) == 3)]
        else:
            output = num_level[num_level['d_level'] == 0]

    output.to_csv(out_dir, index=False, header=None, sep=' ')
    # print(output)
    return output

def combine_all_sds(sds_hospital_dir, sds_lzu_dir, out_dir):
    sds_hospital = pd.read_table(sds_hospital_dir, header=None, delimiter=' ')
    sds_lzu = pd.read_table(sds_lzu_dir, header=None, delimiter=' ')
    combined = pd.concat([sds_hospital,sds_lzu])

    rows = combined.shape[0]
    np.random.seed(50)
    permutation = np.random.permutation(rows)

    shuffled_data = combined.iloc[permutation,:]

    shuffled_data.to_csv(out_dir, index=False, header=None, sep=' ')

def main():
    root_dir = r'H:\Projects\PythonProjects\Graduate_work\010_data\gait\total.csv'
    out_dir = r'H:\Projects\PythonProjects\Graduate_work\010_data\gait\number_sds_phq.csv'
    out_dir1 = r'J:\Projects\PythonProjects\Graduate_work\010_data\gait_ques\number_sds.csv'
    out_dir2 = r'H:\Projects\PythonProjects\Graduate_work\010_data\gait\number_phq.csv'
    depression_dir = r'J:\Projects\PythonProjects\Graduate_work\010_data\gait\depression.xlsx'
    depression_out = r'J:\Projects\PythonProjects\Graduate_work\010_data\gait_ques\depression.csv'
    combine_dir = r'J:\Projects\PythonProjects\Graduate_work\010_data\gait_ques\all_sds_shuffle.csv'
    # separate_sds_phq(out_dir,out_dir1,out_dir2)
    # read_sds_phq(root_dir,out_dir,2)
    # read_excel_p_output(depression_dir, 2, 'd', depression_out)
    combine_all_sds(depression_out,out_dir1,combine_dir)


if __name__ == '__main__':
    main()
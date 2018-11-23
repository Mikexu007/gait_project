import os
import pandas as pd
import numpy as np


_EPS = np.finfo(float).eps * 4.0

def fileOrDir_in_dir(dir_):
    _filesOrDirs = os.listdir(dir_)
    return _filesOrDirs

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path,i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def remove_two_column(root_dir):
    files = fileOrDir_in_dir(root_dir)
    for file in files:
        # sub_dir = os.path.join(root_dir,folder)
        # files = fileOrDir_in_dir(sub_dir)
        # for file in files:
        file_dir = os.path.join(root_dir,file)
        pd_content = pd.read_table(file_dir,header=None,delimiter=' ')
        pd_content = pd_content.iloc[:,2:77]
        pd_content = pd_content.round(8)
        os.remove(file_dir)
        # print(pd_content.shape)
        # name = 'new_'+str(file)
        # new_dir = os.path.join(sub_dir,name)
        pd_content.to_csv(file_dir,index=False, header=None,sep=' ')
        # os.remove(file_dir)

def gaus_filter(data):
    gauf = [0.0625, 0.25, 0.375, 0.25, 0.0625]
    result = np.convolve(data,gauf,'same')
    # result1 = np.convolve(data,gauf,'full')
    # print(result)
    # print(type(result))
    # print(result.shape)
    return result

def gaus_function(root_dir):
    files = fileOrDir_in_dir(root_dir)
    for file in files:
        # sub_dir = os.path.join(root_dir,folder)
        # files = fileOrDir_in_dir(sub_dir)
        # for file in files:
        file_dir = os.path.join(root_dir,file)
        pd_content = pd.read_table(file_dir,header=None,delimiter=' ')
        for column in range(pd_content.shape[1]):
            data = pd_content.iloc[:,column]
            filter_col = gaus_filter(list(data))
            pd_content.iloc[:,column] = filter_col
        pd_content = pd_content.round(8)
        os.remove(file_dir)
        # print(pd_content.shape)
        # name = 'new_'+str(file)
        # new_dir = os.path.join(sub_dir,name)
        pd_content.to_csv(file_dir,index=False, header=None,sep=' ')
        # os.remove(file_dir)

def normalize(root_dir):
    files = fileOrDir_in_dir(root_dir)
    for file in files:
        # sub_dir = os.path.join(root_dir,folder)
        # files = fileOrDir_in_dir(sub_dir)
        # for file in files:
        file_dir = os.path.join(root_dir,file)
        pd_content = pd.read_table(file_dir,header=None,delimiter=' ')
        for row in range(pd_content.shape[0]):
            row_data = pd_content.iloc[row,:]
            new_row = normalize_skeleton(row_data)
            pd_content.iloc[row,:] = new_row
        pd_content = pd_content.round(8)
        os.remove(file_dir)
        # print(pd_content.shape)
        # name = 'new_'+str(file)
        # new_dir = os.path.join(sub_dir,name)
        pd_content.to_csv(file_dir,index=False, header=None,sep=' ')
        # os.remove(file_dir)


def normalize_skeleton(row_data):
    # if anchor == None and norm_dist == 0:
    right_to_left = None
    spine_to_top = None
    anchor = None
    norm_dist = 0
    if norm_dist == 0:
        anchor = np.array([row_data[1*3+0], row_data[1*3+1], row_data[1*3+2]])  # SpineMid
        base = np.array([row_data[0*3+0], row_data[0*3+1], row_data[0*3+2]])  # SpineBase
        norm_dist = np.linalg.norm(anchor - base)

        ## TRANSLATE TO SPINE ORIGIN FIRST
    norm_joints = []
    for jnum in range(int(len(row_data)/3)):
        normalized_pos = np.array([row_data[jnum*3 + 0] - anchor[0],
                                   row_data[jnum*3 + 1] - anchor[1],
                                   row_data[jnum*3 + 2] - anchor[2]])
        norm_joints.extend(normalized_pos)

    if right_to_left is None:
        right_to_left = np.array([norm_joints[8*3+0], norm_joints[8*3+1], norm_joints[8*3+2]]) - np.array(
            [norm_joints[4*3+0], norm_joints[4*3+1], norm_joints[4*3+2]])
        right_to_left = right_to_left / (np.linalg.norm(right_to_left) + _EPS)

    ## COMPUTE ROTATION SUCH THAT RIGHT TO LEFT IS PARALLEL TO X_AXIS
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])

    new_x = right_to_left
    new_y = np.cross(right_to_left, x_axis)
    new_y = new_y / (np.linalg.norm(new_y) + _EPS)
    new_z = np.cross(new_x, new_y)
    new_z = new_z / (np.linalg.norm(new_z) + _EPS)
    Rx = np.transpose(np.array([new_x, new_y, new_z]))
    # pdb.set_trace()

    rotated_and_norm = []
    for jnum in range(int(len(norm_joints)/3)):
        joint = np.array([norm_joints[jnum*3+0],norm_joints[jnum*3+1],norm_joints[jnum*3+2]])
        turn_to_x = np.dot(joint, Rx)
        # turn_to_y = np.dot(Ry,turn_to_x)
        rotated_and_norm.extend(turn_to_x)

    new_right_to_left = np.array([rotated_and_norm[8*3+0], rotated_and_norm[8*3+1], rotated_and_norm[8*3+2]]) - np.array(
        [rotated_and_norm[4*3+0], rotated_and_norm[4*3+1], rotated_and_norm[4*3+2]])

    if spine_to_top is None:
        spine_to_top = np.array([rotated_and_norm[0*3+0], rotated_and_norm[0*3+1], rotated_and_norm[0*3+2]]) - np.array(
            [rotated_and_norm[1*3+0], rotated_and_norm[1*3+1], rotated_and_norm[1*3+2]])
        spine_to_top = spine_to_top / (np.linalg.norm(spine_to_top) + _EPS)

    ## COMPUTE ROTATION SUCH THAT SPINE TO TOP IS PARALLEL TO Y_AXIS
    new_y = spine_to_top
    new_x = np.cross(spine_to_top, y_axis)
    new_x = new_x / (np.linalg.norm(new_x) + _EPS)
    new_z = np.cross(new_x, new_y)
    new_z = new_z / (np.linalg.norm(new_z) + _EPS)
    Ry = np.transpose(np.array([new_x, new_y, new_z]))

    rotated_and_norm2 = []
    for jnum in range(int(len(rotated_and_norm)/3)):
        joint = np.array([rotated_and_norm[jnum*3+0],rotated_and_norm[jnum*3+1],rotated_and_norm[jnum*3+2]])
        # turn_to_x = np.dot(Rx,joint)
        turn_to_y = np.dot(joint, Ry)

        # qR = quaternion_matrix(row_data[jnum][3:])
        # rotated_q = np.dot(np.dot(qR, Rx), Ry)
        # qq = matrix_quaternion(rotated_q)

        normed_final_vec = turn_to_y / (norm_dist + _EPS)

        # print("normed_final_vec shape",normed_final_vec.shape)
        # print("qq shape",qq.shape)
        rotated_and_norm2.extend(normed_final_vec)

    # print(len(rotated_and_norm2))
    rotated_and_norm2 = pd.Series(np.array(rotated_and_norm2))
    rotated_and_norm2 = rotated_and_norm2.round(8)
    # rotated_and_norm2 = list(rotated_and_norm2)
    # print(len(rotated_and_norm2))
    return rotated_and_norm2


def main():
    root_dir = r'J:\Projects\PythonProjects\Graduate_work\010_data\gait_cleaned_data\training_test_data'
    # remove_two_column(root_dir)
    # gaus_function(root_dir)
    normalize(root_dir)



if __name__ == '__main__':
    main()
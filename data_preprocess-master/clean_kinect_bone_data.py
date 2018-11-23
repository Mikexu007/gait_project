#-*- coding: UTF-8 -*-
#!/usr/bin/env python
#
import os
import sys
import optparse
from replay_kinect_bone_data import draw
from matlab_cp2tform import get_similarity_transform_for_cv2
import cv2
import numpy as np
#from builtins import input
try:
   input = raw_input
except NameError:
   pass

JointType_SpineBase = 0
JointType_SpineMid = 1
JointType_Neck = 2
JointType_Head = 3
JointType_ShoulderLeft = 4
JointType_ElbowLeft = 5
JointType_WristLeft = 6
JointType_HandLeft = 7
JointType_ShoulderRight = 8
JointType_ElbowRight = 9
JointType_WristRight = 10
JointType_HandRight = 11
JointType_HipLeft = 12
JointType_KneeLeft = 13
JointType_AnkleLeft=14
JointType_FootLeft = 15
JointType_HipRight = 16
JointType_KneeRight = 17
JointType_AnkleRight=18
JointType_FootRight = 19
JointType_SpineShoulder = 20
JointType_HandTipLeft = 21
JointType_ThumbLeft = 22
JointType_HandTipRight = 23
JointType_ThumbRight = 24
Frame_Num=25

skeletons = []
skeletons_lines = []
value_threshold = 0.09
line_threshold = 6
skeleton_threshold = 6
candidates=[]
op_flag = False
def alignment(src_img, src_pts):
    # ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
    #            [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    ref_pts = [[48.0252, 20.6963],[38,30],[48.0252, 60.7366]]

    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(3, 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    bone_img = cv2.warpAffine(src_img, tfm, crop_size)
    return bone_img
def save_skeleton_video(video_path, start=0, end=10**10, subject_lines=[],num=0):
    write=True
    avi = os.path.join(video_path, "bgr.avi")
    txt = os.path.join(video_path, "kinect color.txt")
    f = open(txt, encoding='utf-8')  # ,encoding='utf-8'
    lines = f.readlines()
    dict = {}
    line_num = 0
    for i in range(len(lines)):
        line_num = line_num + 1
        line = lines[i].split(' ')[0:-1]
        # use color frame as the key in the dict
        if (int(line[1]) not in dict):
            dict[int(line[1])] = []
        values = []
        pre_i = 0
        # there are two frame numbers in a line
        for i in line[2:]:
            try:
                i = float(i)
                values.append(i)
                pre_i = i
            except:
                values.append(pre_i)
        values.append(int(line[0]))
        values.append(line_num)
        dict[int(line[1])].append(values)

    cap = cv2.VideoCapture(avi)
    fg=0#fg是当前帧数
    while 1:
        fg+=1
        ret, copy = cap.read()

        if(fg<start):
            continue
        if(fg>end):
            break
        if (not copy is None):
            if (fg in dict):
                tmp = len(dict[fg])
                skeletons = []
                for j in range(tmp):
                    if (dict[fg][j][Frame_Num * 2] == dict[fg][0][Frame_Num * 2]):
                        jointArr = [[0, 0] for _ in range(26)]
                        for k in range(26):
                            for l in range(2):
                                if l == 0:
                                    jointArr[k][0] = dict[fg][j][k * 2 + l]
                                else:
                                    jointArr[k][1] = dict[fg][j][k * 2 + l]
                        skeletons.append(jointArr)

                if len(skeletons) > 0:
                    bones=[]
                    for jointArr in skeletons:
                        x = []
                        y = []
                        for point in jointArr:
                            x.append(int(point[0]))
                            y.append(int(point[1]))
                        x = sorted(x)
                        y = sorted(y)

                        if jointArr[25][1] in subject_lines:
                            if(x[1]-20<0):
                                x[1]=5
                            if(x[-2]+20>960):
                                x[-2]=954
                            if(y[1]-50<0):
                                y[1]=30
                            if(y[-2]+50>540):
                                y[-2]=509
                            bones.append(jointArr[JointType_Head])
                            bones.append(jointArr[JointType_ShoulderLeft])
                            bones.append(jointArr[JointType_SpineBase])
                            try:
                                copy[:, 0:x[1] - 20] = 0
                                copy[:, x[-2] + 20:960] = 0
                                copy[0:y[1] - 50, :] = 0
                                copy[y[-2] + 50:540, :] = 0
                                res=alignment(copy,bones)
                                res=cv2.flip(res,1)
                                cv2.imshow("replay", res)
                                if write:
                                    dir_name=dest_dir+'/ali_img/'
                                    if not os.path.exists(dir_name):
                                        os.mkdir(dir_name)
                                    dir_name=dir_name+'/'+str(num)
                                    if not os.path.exists(dir_name):
                                        os.mkdir(dir_name)
                                    cv2.imwrite(dir_name+"/"+str(fg)+".jpg",res)
                                    print(dir_name+"/"+str(fg)+".jpg",res)
                            except:
                                continue
            if cv2.waitKey(50) & 0xFF == ord('e'):
                break
    cap.release()
    cv2.destroyAllWindows()

def generate_dest_file_name(src_file):
    path_list = src_file.split(os.sep)
    return "%s_%s" % (path_list[-2], "bone")

def create_destination_dir(dir):
    if not dir:
        print(("The output directory doesn't exist: %s" % dir))
        dir = os.path.join(os.path.dirname(src_file),"cleaned")
        print(("Use the source directory as destination directory: %s" % dir))
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir

def get_patient_dir(filepath):
    path_list = filepath.split(os.sep)
    patient_dir = ""
    try:
        patient_dir = path_list[-4]
    except IndexError:
        pass
    return patient_dir

def load_and_clean_kinect_bone_data(src_file):
    if not os.path.exists(src_file):
        raise SystemExit("The source file doesn't exist : %s" % src_file)
    # read data from file line by line
    line_num = 1
    with open(src_file, "r") as sf:
        for line in sf:
            append_line_to_skeletons(line.strip(), line_num)
            line_num = line_num + 1
    print(("total lines: %d" % line_num))

def append_line_to_skeletons(line, line_num):
    print(("processing the line %d" % line_num))
    values = line.split(' ')
    try:
        v1 = int(values[0])
    except ValueError:
        v1 = float(values[0])
    try:
        v2 = int(values[1])
    except ValueError:
        v2 = float(values[1])

    if not skeletons:
        # the raw data doesn't have frame number
        if not isinstance(v1, int) or v1 == 0:
            values.insert(0, line_num)
        if not isinstance(v2, int) or v2 == 0:
            values.insert(1, line_num)
        print(("insert the line %d to the empty skeletons" % line_num))
        skeletons.append([values])
        skeletons_lines.append([line_num])
    else:
        # the raw data has frame number (After P29)
        if isinstance(v1, int) and v1 != 0:
            # the raw data has 2 frame number (After P40)
            if isinstance(v2, int) and v2 != 0:
                pass
            # the raw data has 1 frame number (P30 to P40)
            else:
                values.insert(1, v1)
        # the raw data doesn't have frame number (P1 to P29)
        else:
            values.insert(0, line_num)
            values.insert(1, line_num)

        processed = False
        for i in range (1, skeleton_threshold + 1):
            try:
                diff_frames = abs(int(skeletons[-i][-1][0]) - int(values[0]))
                if ( diff_frames >=1 and diff_frames <=6 ) and ( abs(float(skeletons[-i][-1][2]) - float(values[2])) <= value_threshold ) :
                    print(("insert the line %d to the skeleton[%d]" % (line_num, len(skeletons)-i+1)))
                    skeletons[-i].append(values)
                    skeletons_lines[-i].append(line_num)
                    processed = True
                    break
            except IndexError as ie:
                print(("the line %d does not match to the last 6 skeletons" % line_num))

        if not processed:
            print(("insert the line %d as a new skeleton" % line_num))
            skeletons.append([values])
            skeletons_lines.append([line_num])
            processed = True

def print_out_skeletons():
    print(("There are %d skeletons in this file" % len(skeletons)))
    i = 0
    global candidates
    for skeleton in skeletons:
        i = i + 1
        if len(skeleton) < 20 or len(skeleton) > 250:
            continue
        candidates.append(i)
        print("===========================================")
        print(("The %d skeleton \n Start from %d to %d, total %d frames(lines)" % (i, int(skeleton[0][0]), int(skeleton[-1][0]), len(skeleton))))
        print((skeleton[0]))
        print("......")
        print((skeleton[-1]))
        print("===========================================")

op_flg = False
def showAvi(dict):
    global candidates
    src=os.path.dirname(src_file)
    num=''
    print(("Skeletons Candidate Set:", candidates))
    if(len(candidates)==0):
        print("Set is Empty.")
        print("## All process completely, Exit")
        exit(0)
    else:
        while(num==''):
            if op_flg:
                num = candidates[0]
            else:
                num = input("Input Skeleton num / e To exit:")
    if (num == "e"):
        exit(0)

    if num != '':
        try:
            num = int(num)
            start = int(skeletons[num - 1][0][1])
            end = int(skeletons[num - 1][-1][1])
            draw(src, start, end, skeletons_lines[num-1],dict)
            confirm = input("Save %d? ((y-->save)/(n-->don't save)/(e-->exit)/(enter or others-->pass)):" % num)
            if confirm == 'y':
                save_skeleton_to_file(num)
                # save_skeleton_video(src,start, end, skeletons_lines[num-1],num)
                candidates.remove(num)
            if confirm=='n':
                candidates.remove(num)
            if confirm == 'e':
                exit(0)
        except ValueError as ve:
           print(("Unknow options [%s] please check your input" % num))
        showAvi(dict)

def print_options_to_user():
    print("Please input the options: (no [])")
    print("     Input [numbers] to save the skeletons: e.g. 1 for the skeleton 1 or 1,2,3 for the skeletons of 1, 2, 3")
    print("     Input [skeletons] or [s] to show the skeletons information: e.g. skeletons or s")
    print("     Input [avi] or [a] to watch movie with bone:e.g. avi or a")
    print("     Input [help] or [h] to show this message again: e.g. help or h")
    print("     Input [exit] or [q] to exit the program: e.g. exit or q")

def handle_user_input(input):
    if input == "skeletons" or input == "s":
        print_out_skeletons()
    elif input == "help" or input == "h":
        print_options_to_user()
    elif input == "e" or input == "q":
        pass
    elif input=="avi" or input=="a":
        showAvi()
    else:
        numbers_str = input.strip().split(',')
        numbers = []
        try:
            for num_str in numbers_str:
                if num_str:
                    numbers.append(int(num_str))
        except ValueError:
            print(("Unknow options [%s] please check your input" % input))
            return

        if len(numbers) > 0:
            for num in numbers:
                save_skeleton_to_file(num)

def save_skeleton_to_file(num):
    try:
        #src_file_name = os.path.splitext(os.path.basename(src_file))[0]
        skeleton_file_name = "%s_%d.txt" % (dest_file_name, num)
        skeleton_file = os.path.join(dest_dir, skeleton_file_name)
        skeletons[num-1]
        with open(skeleton_file, "w") as sf:
            for item in skeletons[num-1]:
                sf.write("%s\n" % " ".join(map(str,item)))
        print(("saved skeleton %d to file: %s" % (num, skeleton_file)))
    except IndexError as ie:
        print(("Failed: %d is not found in skeletons" % num))

#---------------
# Main Script
#---------------
def main():
    usage = "Usage: %prog [options] args"
    parser = optparse.OptionParser(usage)
    parser.add_option("-s", "--source", dest="source", default="", help="The source directory of raw data")
    parser.add_option("-d", "--destination", dest="destination", default="", help="The output directory for cleaned data")
    (opts, extra) = parser.parse_args()

    # Verify the arguments
    if len(sys.argv) == 1:
        parser.error("No argument has been provided. \nTry --help to get more details.")

    if not os.path.exists(opts.source):
        raise SystemExit("The given source file doesn't exist : %s" % opts.source)

    kinect_bone_file = os.path.join(opts.source, "kinect bone.txt")
    kinect_bgr_file = os.path.join(opts.source, "bgr.avi")

    if not os.path.isfile(kinect_bone_file):
        raise SystemExit("The file 'kinect bone.txt' doesn't exist in : %s" % opts.source)
    if not os.path.isfile(kinect_bgr_file):
        raise SystemExit("The file 'bgr.avi' doesn't exist in : %s" % opts.source)

    global src_file
    src_file = kinect_bone_file
    global dest_file_name
    dest_file_name = generate_dest_file_name(src_file)
    global dest_dir
    dest_dir = create_destination_dir(opts.destination)

    load_and_clean_kinect_bone_data(kinect_bone_file) # save cleaned to skeletons

    print_out_skeletons()

    inputs = ""
    print_options_to_user()
    # 直接进入符合骨架条件的视频

    '''
    copy from replay
    '''
    video_path = os.path.dirname(src_file)
    if not os.path.exists(video_path):
        raise SystemExit("The video path doesn't exist : %s" % video_path)
    avi = os.path.join(video_path, "bgr.avi")
    txt = os.path.join(video_path, "kinect color.txt")
    f=open(txt, encoding='utf-8') #,encoding='utf-8'
    lines=f.readlines()
    dict={}
    line_num = 0
    for i in range(len(lines)):
        line_num = line_num + 1
        line=lines[i].split(' ')[0:-1]
        # use color frame as the key in the dict
        if(int(line[1]) not in dict):
            dict[int(line[1])]=[]
        values = []
        pre_i = 0
        # there are two frame numbers in a line
        for i in line[2:]:
            try:
                i = float(i)
                values.append(i)
                pre_i = i
            except:
                values.append(pre_i)
        values.append(int(line[0]))
        values.append(line_num)
        dict[int(line[1])].append(values)
    '''
    end
    '''

    showAvi(dict)
    while inputs != "exit" and inputs != "q":
        inputs = input("Enter your option: ")
        if inputs:
            handle_user_input(inputs)
        else:
            print("Nothing has been input")

    print("Exit the program.")


if __name__ == '__main__':
    main()

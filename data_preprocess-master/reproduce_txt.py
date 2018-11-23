import os
from replay_kinect_bone_data import draw
import numpy as np
import cv2

JointType_SpineBase=0
JointType_SpineMid=1
JointType_Neck=2
JointType_Head=3
JointType_ShoulderLeft=4
JointType_ElbowLeft=5
JointType_WristLeft=6
JointType_HandLeft=7
JointType_ShoulderRight=8
JointType_ElbowRight=9
JointType_WristRight=10
JointType_HandRight=11
JointType_HipLeft=12
JointType_KneeLeft=13
JointType_AnkleLeft=14
JointType_FootLeft=15
JointType_HipRight=16
JointType_KneeRight=17
JointType_AnkleRight=18
JointType_FootRight=19
JointType_SpineShoulder=20
JointType_HandTipLeft=21
JointType_ThumbLeft=22
JointType_HandTipRight=23
JointType_ThumbRight=24

def reproduce_txt(src_file = r"F:\python_projects\data\\raw_data\h\h1\8_kinects\\CollectorA1_08-15_09-38_0",
         cleaned_txt="F:/python_projects/data/kinect_bone_cleaned/h1/CollectorA1_08-15_09-38_0_bone_1.txt"):
    video_path = os.path.join(src_file, 'bgr.avi')
    if not os.path.exists(video_path):
        raise SystemExit("The video path doesn't exist : %s" % video_path)
    txt = os.path.join(src_file, "kinect color.txt")
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
    f = open(cleaned_txt)
    clean_lines = f.readlines()
    clean_lines = [line.strip() for line in clean_lines]
    clean_lines = [line.split(' ') for line in clean_lines]
    clean_lines = np.array(clean_lines).astype(float).tolist()

    start_frame_number = clean_lines[0][1]
    end_frame_number = clean_lines[-1][1]
    draw(src_file, start_frame_number, end_frame_number, clean_lines, dict)


def _draw(img, p_1, p_2, color):
    p_1 = (int(p_1[0]), int(p_1[1]))
    p_2 = (int(p_2[0]), int(p_2[1]))
    cv2.line(img, p_1, p_2, color, 3)
    cv2.circle(img, p_1, 5, (255, 0, 0), -1)
    cv2.circle(img, p_2, 5, (255, 0, 0), -1)
    return img


def draw_skeleton(copy, jointArr, color):
    copy =_draw(copy, jointArr[JointType_Head], jointArr[JointType_Neck], color)
    copy =_draw(copy, jointArr[JointType_Neck], jointArr[JointType_SpineShoulder], color)

    copy =_draw(copy, jointArr[JointType_SpineShoulder], jointArr[JointType_ShoulderLeft], color)
    copy =_draw(copy, jointArr[JointType_SpineShoulder], jointArr[JointType_SpineMid], color)
    copy = _draw(copy, jointArr[JointType_SpineShoulder], jointArr[JointType_ShoulderRight], color)

    copy =_draw(copy, jointArr[JointType_ShoulderLeft], jointArr[JointType_ElbowLeft], color)
    copy =_draw(copy, jointArr[JointType_SpineMid], jointArr[JointType_SpineBase], color)
    copy =_draw(copy, jointArr[JointType_ShoulderRight], jointArr[JointType_ElbowRight], color)

    copy =_draw(copy, jointArr[JointType_ElbowLeft], jointArr[JointType_WristLeft], color)
    copy =_draw(copy, jointArr[JointType_SpineBase], jointArr[JointType_HipLeft], color)
    copy =_draw(copy, jointArr[JointType_SpineBase], jointArr[JointType_HipRight], color)
    copy =_draw(copy, jointArr[JointType_ElbowRight], jointArr[JointType_WristRight], color)

    copy =_draw(copy, jointArr[JointType_WristLeft], jointArr[JointType_ThumbLeft], color)
    copy =_draw(copy, jointArr[JointType_WristLeft], jointArr[JointType_HandLeft], color)
    copy =_draw(copy, jointArr[JointType_HipLeft], jointArr[JointType_KneeLeft], color)
    copy =_draw(copy, jointArr[JointType_HipRight], jointArr[JointType_KneeRight], color)
    copy =_draw(copy, jointArr[JointType_WristRight], jointArr[JointType_ThumbRight], color)
    copy =_draw(copy, jointArr[JointType_WristRight], jointArr[JointType_HandRight], color)

    copy =_draw(copy, jointArr[JointType_HandLeft], jointArr[JointType_HandTipLeft], color)
    copy =_draw(copy, jointArr[JointType_KneeLeft], jointArr[JointType_FootLeft], color)
    copy =_draw(copy, jointArr[JointType_KneeRight], jointArr[JointType_FootRight], color)
    copy =_draw(copy, jointArr[JointType_HandRight], jointArr[JointType_HandTipRight], color)

    return copy



def draw_on_a_black_background(src_file = r"F:\python_projects\data\\raw_data\h\h1\8_kinects\\CollectorA1_08-15_09-38_0",
         cleaned_txt="F:/python_projects/data/kinect_bone_cleaned/h1/CollectorA1_08-15_09-38_0_bone_1.txt"):
    video_path = os.path.join(src_file, 'bgr.avi')
    if not os.path.exists(video_path):
        raise SystemExit("The video path doesn't exist : %s" % video_path)
    txt = os.path.join(src_file, "kinect color.txt")
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
    f = open(cleaned_txt)
    clean_lines = f.readlines()
    clean_lines = [line.strip() for line in clean_lines]
    clean_lines = [line.split(' ') for line in clean_lines]
    clean_lines = np.array(clean_lines).astype(float).tolist()

    fg=0
    subject_lines = clean_lines
    start = clean_lines[0][1]
    end = clean_lines[-1][1]
    Frame_Num = 25
    while(1):
        fg+=1
        copy = np.ones((540, 960, 3), dtype=np.uint8)
        copy[:, :, :] = 0
        if(fg<start):
            continue
        if(fg>end):
            break

        if(not copy is None):
            if(fg in dict):
                tmp=len(dict[fg])
                skeletons = []
                for j in range(tmp):
                    if (dict[fg][j][Frame_Num*2] == dict[fg][0][Frame_Num*2]):
                        jointArr = [[0,0] for _ in range(26)]
                        for k in range(26):
                            for l in range(2):
                                if l==0:
                                    jointArr[k][0]=dict[fg][j][k*2+l]
                                else:
                                    jointArr[k][1]=dict[fg][j][k*2+l]
                        skeletons.append(jointArr)

                if len(skeletons) > 0:
                    for jointArr in skeletons:
                        x=[]
                        y=[]
                        for index,point in enumerate(jointArr):
                            #if index==14 or index==18:
                            #    continue
                            x.append(int(point[0]))
                            y.append(int(point[1]))
                        x=sorted(x)
                        y=sorted(y)
                        if jointArr[25][1] in subject_lines:
                            if(x[1]-5<0):
                                x[1]=5
                            if(x[-2]+5>960):
                                x[-2]=954
                            if(y[1]-30<0):
                                y[1]=30
                            if(y[-2]+50>540):
                                y[-2]=489
                            color = (0, 0, 255)
                            #0->x[1]-5           x[-2]+5->960          0->y[1]-30            y[-2]+30->540
                            #copy[0:x[1]-5,:]    copy[x[-2]+5:960,:]   copy[:,0:y[1]-30]     copy[:,y[-2]+30:540]
                            # copy[:,0:x[1] - 5]=0
                            # copy[ :,x[-2] + 5:960]=0
                            # copy[0:y[1] - 30,:]=0
                            # copy[y[-2] + 30:540,:]=0
                            cv2.rectangle(copy,(x[1]-5,y[1]-30),(x[-2]+5,y[-2]+50),(0,0,255))
                        else:
                            color = (0, 255, 0)
                        copy = draw_skeleton(copy, jointArr, color)
            cv2.imshow("replay",copy)
            key = cv2.waitKey(42) & 0xFF
            if key == ord('e') or key == 27:
                break
        else:
            break
    cv2.destroyAllWindows()



if __name__=="__main__":
    reproduce_txt("F:\python_projects\data\\raw_data\h\h1\8_kinects\\CollectorA1_08-15_09-38_0",
         'F:/python_projects/data/kinect_bone_cleaned/h1/CollectorA1_08-15_09-38_0_bone_1.txt')
    draw_on_a_black_background("F:\python_projects\data\\raw_data\h\h1\8_kinects\\CollectorA1_08-15_09-38_0",
         'F:/python_projects/data/kinect_bone_cleaned/h1/CollectorA1_08-15_09-38_0_bone_1.txt')
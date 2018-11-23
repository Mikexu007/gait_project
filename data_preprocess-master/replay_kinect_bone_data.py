#-*- coding: UTF-8 -*-
#!/usr/bin/env python
#
import os
import cv2
import sys
import optparse

# define the structure of kinect data in the data file
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
Frame_Num=25  # depth, line_num

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

def draw(video_path, start=0, end=10**10, subject_lines=[],dict={}):
    avi = os.path.join(video_path, "bgr.avi")
    cap=cv2.VideoCapture(avi)
    fg=0
    exit_flag = False
    isPlay = True

    while(1):
        fg+=1
        ret, copy = cap.read()
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
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':

    usage = "Usage: %prog [options] args"
    parser = optparse.OptionParser(usage)
    parser.add_option("-s", "--source", dest="source", default="", help="The source directory with kinect video and bone data")
    parser.add_option("-d", "--destination", dest="destination", default="", help="The output directory for cleaned data")
    (opts, extra) = parser.parse_args()

    # Verify the arguments
    if len(sys.argv) == 1:
        parser.error("No argument has been provided. \nTry --help to get more details.")

    if not os.path.exists(opts.source):
        raise SystemExit("The given source directory doesn't exist : %s" % opts.source)
    video_path = opts.source

    draw(video_path)

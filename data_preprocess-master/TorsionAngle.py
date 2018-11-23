# -*- coding: utf-8 -*-
"""
Created on 2018/11/10 14:08
@author: Eric
@email: qian.dong.2018@gmail.com
"""
import os
import math
def torsionAngle(filename, z_offset=0.0, reverse=False, angle=27.0):
    f = open(filename)
    lines = f.readlines()
    f.close()
    lines = [line.strip().split(' ') for line in lines]
    new_lines=[]
    for line in lines:
        if len(line) == 1:
            continue
        new_line=[]
        for i in range(25):
            idx=i*3
            pre_x = 0
            pre_y = 0
            pre_z = 0
            try:
                _x = float(line[2:][idx])
                _y = float(line[2:][idx+1])
                _z = float(line[2:][idx+2])
                pre_x = float(line[2:][idx])
                pre_y = float(line[2:][idx+1])
                pre_z = float(line[2:][idx+2])
            except:
                _x = pre_x
                _y = pre_y
                _z = pre_z
            a = _z*math.tan(angle/180*math.pi)
            b = _y-a
            c = _z/math.cos(angle/180*math.pi)
            d = b*math.sin(angle/180*math.pi)
            y = b*math.cos(angle/180*math.pi)+2.6
            z = c+b*math.sin(angle/180*math.pi)
            x = _x
            if reverse:
                x = -x
                z = -z
            z = z+z_offset
            if i == 0:
                new_line.append(int(line[0]))
                new_line.append(int(line[1]))
            new_line.append(x)
            new_line.append(y)
            new_line.append(z)
        new_lines.append(new_line)
    return new_lines

if __name__=="__main__":
    out_path_root="../data/kinect_bone_cleaned_and_affine_transformed"
    for root, dirs, files in os.walk("../data/kinect_bone_cleaned/"):
        for name in files:
            if "txt" in name:
                out_name = os.path.join(out_path_root, root.split('/')[-1], name).replace('\\', '/')
                name = os.path.join(root, name).replace('\\', '/')
                print(name)
                if "CollectorA1" in name:
                    new_lines = torsionAngle(filename=name, z_offset=-1.5, reverse=False)
                elif "CollectorB2" in name:
                    new_lines = torsionAngle(filename=name, z_offset=1.0, reverse=False)
                elif "CollectorC3" in name:
                    new_lines = torsionAngle(filename=name, z_offset=3.7, reverse=True)
                elif "CollectorD4" in name:
                    new_lines = torsionAngle(filename=name, z_offset=3.8, reverse=False)
                elif "CollectorE5" in name:
                    new_lines = torsionAngle(filename=name, z_offset=6.2, reverse=True)
                elif "CollectorF6" in name:
                    new_lines = torsionAngle(filename=name, z_offset=6.3, reverse=False)
                elif "CollectorG7" in name:
                    new_lines = torsionAngle(filename=name, z_offset=9.0, reverse=True)
                elif "CollectorH8" in name:
                    new_lines = torsionAngle(filename=name, z_offset=11.5, reverse=True)
                else:
                    continue
                new_lines = [str(line).replace(',', '').replace('[', '').replace(']', '') + '\n' for line in new_lines]
                if not os.path.exists(os.path.dirname(out_name)):
                    os.makedirs(os.path.dirname(out_name))
                with open(out_name, 'w')as out:
                    for line in new_lines:
                        out.write(line)
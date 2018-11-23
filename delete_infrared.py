#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

def remove_infraed(root_dir):
    list_dir = os.listdir(root_dir)
    for collector in list_dir:
        splits = collector.split('_')
        if len(splits) >= 3:
            date = splits[2]
            day = date.split("-")[1]
            if day == '26':
                sub_dir = os.path.join(root_dir,collector)
                target_infraed = os.path.join(sub_dir,"infrared.zip")
                if os.path.exists(target_infraed):
                    os.remove(target_infraed)


root_dir = r'I:\gait_nas'
remove_infraed(root_dir)
# Applies degradation to all images in a folder.

from os import listdir, path
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import configparser
from cv2 import cv2  # fixes linting bug
import numpy as np
from vanishing_point_detection import get_vp
from road_area_detection import get_roi
from road_marking_detection import get_rm
import degrade_rm
import sys
sys.path.insert(0, '.\\line_detection')
sys.path.insert(0, '.\\line_degradation')


# setup config and logging
config = configparser.ConfigParser()
config.read('settings\\oxford.ini')
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# get image file names
# src_dir = Path('C:\\Users\\Georg\\Desktop\\datasets\\KITTI\\2011_09_26_drive_0001_extract\\2011_09_26\\2011_09_26_drive_0001_extract\\image_02\\data')
src_dir = Path(
    'C:\\Users\\Georg\\Desktop\\datasets\\oxford\\2015-10-29-12-18-17_stereo_centre_07\\demosaiced')
dst_dir = Path('C:\\Users\\Georg\\Desktop\\datasets\\temp')
files = [f for f in listdir(src_dir) if path.isfile(
    path.join(src_dir, f))]
files = files[100:120]

# read images
img_array = []
for filename in files:
    img = cv2.imread(str(src_dir/filename))
    img_array.append(img)

# process images
for i in range(len(img_array)):
    vp = get_vp(img_array[i], config)
    roi = get_roi(img_array[i], vp, config)
    mask = get_rm(img_array[i], vp, roi, config)
    img_output = degrade_rm.degrade_rm(img_array[i], mask, config)
    cv2.imwrite(str(dst_dir) + '\\' + files[i], img_output)

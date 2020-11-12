# Demosaic images from oxford roborcar dataset

import numpy as np
from cv2 import cv2
from pathlib import Path
from os import listdir, path

import image

src_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\oxford\\2015-02-20-16-34-06_stereo_centre_07\\2015-02-20-16-34-06\\stereo\\centre'
dst_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\oxford\\2015-02-20-16-34-06_stereo_centre_07\\demosaic'

files = [f for f in listdir(src_dir) if path.isfile(path.join(src_dir, f))]

for image_name in files:
    image_path = src_dir + '\\' + image_name
    image_demosaic = image.load_image(image_path)
    cv2.imwrite(dst_dir + '\\' + image_name, image_demosaic[:, :, [2, 1, 0]])

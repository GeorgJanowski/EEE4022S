# Creates images to be used in pix2pix GAN

import sys
sys.path.insert(0, '.\\line_detection')
sys.path.insert(0, '.\\line_degradation')

import degrade_rm
from road_marking_detection import get_rm
# from road_area_detection import get_roi
# from vanishing_point_detection import get_vp
from oxford_rm import get_oxford_rm
import numpy as np
from cv2 import cv2  # fixes linting bug
import configparser
import logging
import matplotlib.pyplot as plt
import glob
from os import path

# 382 1447 3165


# setup config and logging
config = configparser.ConfigParser()
config.read('settings\\oxford.ini')
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')

src_dir = config['data']['src_dir']
dst_dir = config['data']['dst_dir']
file_paths = glob.glob(src_dir + '\\*.png')
file_paths = file_paths[::10]
print(len(file_paths))


vp_x = config['vp'].getint('default_x')
vp_y = config['vp'].getint('default_y')
vp = (vp_x, vp_y)

roi_mask_path = './settings/oxford_mask.png'
roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)

counter = 1
for file_path in file_paths:
    file_name = path.basename(file_path)[:-3] + 'jpg'
    print(counter, file_name)
    img_input = cv2.imread(file_path)

    rm_mask_adaptive = get_oxford_rm(img_input, roi_mask)
    rm_mask_otsu = get_rm(img_input, vp, roi_mask, config)
    rm_mask = cv2.bitwise_and(rm_mask_adaptive, rm_mask_adaptive, mask=rm_mask_otsu)
    img_output = degrade_rm.degrade_rm(img_input, rm_mask, config)

    img_output = cv2.resize(img_output, (256,256))
    img_input = cv2.resize(img_input, (256,256))

    example = np.concatenate((img_output, img_input), axis=1)
    cv2.imwrite(dst_dir + '\\' + file_name, example)

    counter += 1

    # DEBUG
    # cv2.imwrite('temp/img_input_' + file_name, img_input)
    # cv2.imwrite('temp/rm_mask_adaptive_' + file_name, rm_mask_adaptive)
    # cv2.imwrite('temp/rm_mask_otsu_' + file_name, rm_mask_otsu)
    # cv2.imwrite('temp/rm_mask_' + file_name, rm_mask)
    # cv2.imwrite('temp/img_output_' + file_name, img_output)

    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(img_input[:,:,[2,1,0]])
    # plt.title('input')
    # plt.subplot(2,2,2)
    # plt.imshow(rm_mask_adaptive, cmap='gray')
    # plt.title('adaptive')
    # plt.subplot(2,2,3)
    # plt.imshow(rm_mask_otsu, cmap='gray')
    # plt.title('otsu')
    # plt.subplot(2,2,4)
    # plt.imshow(rm_mask, cmap='gray')
    # plt.title('combined')
    # plt.show()
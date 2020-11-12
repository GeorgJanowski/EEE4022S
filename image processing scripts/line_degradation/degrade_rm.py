# Degrade detected line markings in different ways

from add_pnoise import generate_pnoise
from vanishing_point_detection import get_vp
from road_area_detection import get_roi
from road_marking_detection import get_rm
import logging
import numpy as np
from cv2 import cv2  # fixes linting bug

import configparser
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir, path
import sys
sys.path.insert(0, '.\\line_detection')


def remove_rm(img, mask, config):
    """
    Remove detected road markings entirely by inpainting.
    Dilate road marking mask to ensure painted regions are completely covered.
    """
    dilate_size = config['degrade'].getint('dilate_size')
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    mask = cv2.dilate(mask, kernel)
    img_inpaint = cv2.inpaint(img, mask, 15, cv2.INPAINT_TELEA)
    return img_inpaint


def remove_part_rm(img, rm_mask, config):
    """
    Remove parts of detected road markings by overlaying the original image and
    an image which has had all road markings removed.
    The two images are overlayed using binarised perlin noise.
    """
    img_inpaint = remove_rm(img, rm_mask, config)
    pnoise = generate_pnoise(img.shape)
    pnoise = cv2.cvtColor(pnoise, cv2.COLOR_BGR2GRAY)
    _, pnoise_thr = cv2.threshold(pnoise, 120, 255, cv2.THRESH_BINARY)
    rm_pnoise_mask = cv2.bitwise_and(rm_mask, rm_mask, mask=pnoise_thr)
    rm_pnoise_mask = cv2.dilate(rm_pnoise_mask, np.ones((3, 3), np.uint8))
    rm_pnoise_mask_inv = cv2.bitwise_not(rm_pnoise_mask)
    img_inpaint_masked = cv2.bitwise_and(
        img_inpaint, img_inpaint, mask=rm_pnoise_mask)
    img_masked = cv2.bitwise_and(img, img, mask=rm_pnoise_mask_inv)
    img_output = cv2.add(img_masked, img_inpaint_masked)

    # DEBUG
    cv2.imshow('noise_mask', pnoise_thr)
    cv2.waitKey(0)
    cv2.imwrite('temp/pnoise_thr.png', pnoise_thr)
    # rm_degrade = cv2.addWeighted(rm_inpaint, 0.5, pnoise, 0.5, 0)
    # print(rm_degrade.shape, rm_degrade.dtype)
    # cv2.imshow('rm_degrade', rm_degrade)
    # cv2.waitKey(0)

    return img_output


def wear_rm(img, rm_mask, config):
    """
    Wears detected road markings by overlaying the original image and an image
    which has had all road markings removed.
    The two images are overlayed using perlin noise.
    """
    img_inpaint = remove_rm(img, rm_mask, config)
    pnoise = generate_pnoise(img.shape)
    unoise = np.empty(img.shape[:2], np.uint8)
    cv2.randu(unoise, 0, 255)
    unoise = cv2.cvtColor(unoise, cv2.COLOR_GRAY2BGR)
    noise = cv2.addWeighted(pnoise, 1.5, unoise, 0.5, -200)
    noise_inv = cv2.bitwise_not(noise)
    img_overlay = cv2.multiply(img.astype(
        np.float32), noise.astype(np.float32) / 255)
    img_overlay = img_overlay.astype(np.uint8)
    img_inpaint_overlay = cv2.multiply(img_inpaint.astype(
        np.float32), noise_inv.astype(np.float32) / 255)
    img_inpaint_overlay = img_inpaint_overlay.astype(np.uint8)
    img_output = cv2.addWeighted(img_overlay, 1, img_inpaint_overlay, 1, 0)

    return img_output


def degrade_rm(img, rm_mask, config):
    degrade_type = config['degrade'].getint('type')
    if degrade_type == 0:
        img_output = remove_rm(img, rm_mask, config)
    elif degrade_type == 1:
        img_output = remove_part_rm(img, rm_mask, config)
    elif degrade_type == 2:
        img_output = wear_rm(img, rm_mask, config)
    else:
        logging.error('Invalid degrade type.')
        img_output = None
    return img_output


def main():
    # setup config and logging
    config = configparser.ConfigParser()
    config.read('settings\\kitti.ini')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # load image
    src_dir = Path(config['data']['src_dir'])
    if config['data'].getboolean('use_num'):
        files = [f for f in listdir(
            src_dir) if path.isfile(path.join(src_dir, f))]
        img_path = str(src_dir/files[config['data'].getint('file_num')])
    else:
        img_path = str(src_dir/config['data']['file_name'])
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # get mask of road markings
    vp = get_vp(img, config)
    roi = get_roi(img, vp, config)
    mask = get_rm(img, vp, roi, config)

    # apply degradation
    img_output = degrade_rm(img, mask, config)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('img')
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.subplot(3, 1, 2)
    plt.title('mask')
    plt.imshow(mask, cmap='gray')
    plt.subplot(3, 1, 3)
    plt.title('img_output')
    plt.imshow(img_output[:, :, [2, 1, 0]])

    plt.figure()
    plt.title('img_output')
    plt.imshow(img_output[:, :, [2, 1, 0]])
    plt.show()

    # cv2.imwrite('temp/11.png', img)
    # cv2.imwrite('temp/12.png', img_output)


if __name__ == '__main__':
    main()

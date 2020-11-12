# Detect road markings on image given a mask of the road area
# - detect road markings in sunlight using a grayscale threshold on roi
# - detect areas of shadow using saturation threshold
# - detect road markings in shadow using a grayscale threshold

import numpy as np
from cv2 import cv2  # fixes linting bug

import logging
import configparser
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir, path

from vanishing_point_detection import get_vp
from road_area_detection import get_roi
from road_from_semantics import binarise_semantics
from masked_otsu_threshold import masked_otsu_threshold


def get_shadow(img, roi):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(img_hsv)
    S_blur = cv2.GaussianBlur(S, (3, 3), 0)
    thr_shadow = masked_otsu_threshold(S_blur, roi)
    _, img_shadow = cv2.threshold(S_blur, thr_shadow, 255, cv2.THRESH_BINARY)
    # mask shadows with roi
    roi_shadow = cv2.bitwise_and(img_shadow, img_shadow, mask=roi)
    # get rid of noise outside shadows
    kernel_open = np.array([[0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0]], np.uint8)
    roi_shadow = cv2.morphologyEx(
        roi_shadow, cv2.MORPH_OPEN, kernel_open, iterations=1)
    # get rid of noise in shadows
    kernel_close = np.ones((3, 3), np.uint8)
    roi_shadow = cv2.morphologyEx(
        roi_shadow, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    # erode shadows slightly
    kernel_erode = np.ones((2, 2), np.uint8)
    roi_shadow = cv2.erode(roi_shadow, kernel_erode)

    return roi_shadow


def rm_otsu_sunshade(img, roi, config):
    shadow = get_shadow(img, roi)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # get threshold for shadow region
    # print('thr_in_shadow')
    thr_in_shadow = masked_otsu_threshold(img_gray, shadow)
    # get road markings in shadow
    _, rm_in_shadow = cv2.threshold(
        img_gray, thr_in_shadow, 255, cv2.THRESH_BINARY)
    # kernel_erode = np.ones((7,7), np.uint8)
    # shadow_eroded = cv2.erode(roi_shadow, kernel_erode)
    rm_in_shadow = cv2.bitwise_and(rm_in_shadow, rm_in_shadow, mask=shadow)
    # get threshold for sunlight region
    shadow_inv = cv2.bitwise_not(shadow)
    shadow_inv = cv2.bitwise_and(shadow_inv, shadow_inv, mask=roi)
    thr_out_shadow = masked_otsu_threshold(img_gray, shadow_inv)
    # get road markings not in shadow
    _, rm_out_shadow = cv2.threshold(img_gray, int(
        (thr_out_shadow*1.5) % 255), 255, cv2.THRESH_BINARY)
    # rm_out_shadow = cv2.bitwise_and(rm_out_shadow, rm_out_shadow, mask=shadow_inv)
    # combine markings in shadow and not in shadow
    rm = cv2.bitwise_or(rm_in_shadow, rm_out_shadow)
    rm = cv2.bitwise_and(rm, rm, mask=roi)

    return rm


def get_rm_sunlight(img, roi):
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    # img = cv2.GaussianBlur(img,(5,5),0)
    # simple thresholding
    _, rm_mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # mask roi
    rm_mask = cv2.bitwise_and(rm_mask, roi)
    return rm_mask


def get_rm_shadow(img, roi):
    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Gaussian blur
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    # get saturation
    _, S, _ = cv2.split(hsv)
    # threshold saturation, shadows have higher sat than road
    _, shadow_mask = cv2.threshold(S, 70, 255, cv2.THRESH_BINARY)
    # get rid of noise
    kernel = np.ones((7, 7), np.uint8)
    shadow_mask = cv2.morphologyEx(
        shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # mask roi
    shadow_mask = cv2.bitwise_and(shadow_mask, roi)
    # get shadows from img
    shadows = cv2.bitwise_and(img, img, mask=shadow_mask)
    # make shadows grayscale
    shadows_gray = cv2.cvtColor(shadows, cv2.COLOR_BGR2GRAY)
    # get road markings in shadow
    _, rm_mask = cv2.threshold(shadows_gray, 100, 255, cv2.THRESH_BINARY)
    # clean up noise
    kernel = np.ones((2, 2), np.uint8)
    rm_mask = cv2.morphologyEx(rm_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # increase mask size a bit
    kernel = np.ones((3, 3), np.uint8)
    rm_mask = cv2.dilate(rm_mask, kernel)

    return rm_mask


def rm_static_noshade(img, roi, config):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thr = config['rm'].getint('static_noshade_thr')
    _, rm_mask = cv2.threshold(img_blur, thr, 255, cv2.THRESH_BINARY)
    rm_mask = cv2.bitwise_and(rm_mask, roi)
    return rm_mask


def rm_static_sunshade(img, roi, config):
    sunlight_mask = get_rm_sunlight(img, roi)
    shadow_mask = get_rm_shadow(img, roi)
    rm_mask = cv2.bitwise_or(sunlight_mask, shadow_mask)
    return rm_mask


def rm_otsu_noshade(img, roi, config):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr_otsu = masked_otsu_threshold(img_gray, roi)
    thr_offset = config['rm'].getint('otsu_noshade_offset')
    _, rm_mask = cv2.threshold(
        img_gray, thr_otsu + thr_offset, 255, cv2.THRESH_BINARY)
    rm_mask = cv2.bitwise_and(rm_mask, roi)
    return rm_mask


def get_rm(img, vp, roi, config):
    rm_type = config['rm'].getint('type')
    if rm_type == 0:
        rm_mask = rm_static_noshade(img, roi, config)
    elif rm_type == 1:
        rm_mask = rm_static_sunshade(img, roi, config)
    elif rm_type == 2:
        rm_mask = rm_otsu_noshade(img, roi, config)
    elif rm_type == 3:
        rm_mask = rm_otsu_sunshade(img, roi, config)
    else:
        logging.error('Invalid road marking type.')
        rm_mask = None

    # remove glare near vp
    if config['rm'].getboolean('has_glare'):
        # cv2.imshow('before deglare', rm_mask)
        # cv2.waitKey(0)
        radius = config['rm'].getint('glare_radius')
        rm_mask = cv2.circle(rm_mask, vp, radius, 255, cv2.FILLED)
        # cv2.imshow('before deglare', rm_mask)
        # cv2.waitKey(0)
        cv2.floodFill(rm_mask, None, vp, 0)
        # cv2.imshow('before deglare', rm_mask)
        # cv2.waitKey(0)

    return rm_mask


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
    img = cv2.imread(img_path)

    vp = get_vp(img, config)
    roi = get_roi(img, vp, config)
    rm = get_rm(img, vp, roi, config)
    road_markings = cv2.bitwise_and(img, img, mask=rm)

    # display results
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(roi, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(rm, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(road_markings[:, :, [2, 1, 0]])
    plt.axis('off')


if __name__ == '__main__':
    main()

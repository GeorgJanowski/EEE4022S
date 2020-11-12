# Detect region of interest (roi) where road is estimated to be.
# - use saturation with threshold to determine which parts of image are gray
# - use vanishing point to ignore anything above the horizon
# - use morphological close and open to clean up image
# - integrate binary image to smooth out areas of road that are not detected
# - binarise to get roi mask


import numpy as np
from cv2 import cv2  # fix linter bug

import logging
import configparser
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir, path

from vanishing_point_detection import get_vp


def detect_saturation(img, vp, config):
    # get thresholded saturation
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, S, _ = cv2.split(img_hsv)
    kernel_size = config['roi'].getint('kernel_size')
    S = cv2.GaussianBlur(S, (kernel_size, kernel_size), 0)
    _, S_threshold = cv2.threshold(S, config['roi'].getint(
        'saturation_thr'), 255, cv2.THRESH_BINARY_INV)
    # mask out sky using vp
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array(
        [[(0, vp[1]), (width, vp[1]), (width, height), (0, height)]])
    mask = np.zeros_like(S_threshold)
    cv2.fillPoly(mask, vertices, 255)
    S_threshold_masked = cv2.bitwise_and(S_threshold, mask)
    return S_threshold_masked


def detect_dummy(img, vp, config):
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height), (0, int(height * 2/3)),
                          (vp[0], vp[1]), (width, int(height * 2/3)), (width, height)]])
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def get_roi(img, vp, config):
    road_type = config['roi'].getint('type')
    if road_type == 0:
        road = detect_saturation(img, vp, config)
    elif road_type == 1:
        road = detect_dummy(img, vp, config)
    else:
        logging.error('Unrecognised road type.')

    # morphological close - fill in noise close to vp
    morph_close_size = config['roi'].getint('morph_close_size')
    kernel = np.ones((morph_close_size, morph_close_size), np.uint8)
    closed = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)

    # created circular kernel for morph open
    n = config['roi'].getint('morph_open_size')
    kernel_open = np.ones((n, n), np.uint8)
    for x in range(n):
        for y in range(n):
            kernel_open[x, y] = ((x - n/2) ** 2 + (y - n/2)
                                 ** 2) <= ((n * n) / 4)

    # morphological open - get rid of outlying noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # integral tilted by 45 degrees - fill in foreground shadows and noise
    _, _, integral_tilted = cv2.integral3(opened)

    # binarise to use as roi mask
    roi = ((integral_tilted[:-1, :-1] > 10) * 255).astype(np.uint8)

    if config['roi'].getboolean('has_frame'):
        frame_path = config['roi']['frame_path']
        frame = cv2.imread(frame_path)[:, :, 0]  # only one channel needed
        roi = cv2.bitwise_and(roi, frame)

    return roi


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

    vp = get_vp(img, config)

    roi = get_roi(img, vp, config)

    # mask roi
    img_roi = cv2.bitwise_and(img, img, mask=roi)

    # display results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.subplot(2, 1, 2)
    plt.imshow(img_roi[:, :, [2, 1, 0]])
    plt.show()

    plt.figure()
    plt.imshow(img_roi[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()

    # cv2.imwrite('temp/kitti_roi_detection.png', img_roi)


if __name__ == '__main__':
    main()

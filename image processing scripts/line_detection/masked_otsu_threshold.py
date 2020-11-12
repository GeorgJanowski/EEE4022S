# Find Otsu binarisation threshold of masked image.
# Based on Otsu binarisation example found at:
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html


import logging
import configparser
from cv2 import cv2  # fix linter bug
import numpy as np
from pathlib import Path
from os import listdir, path

from road_area_detection import get_roi
from vanishing_point_detection import get_vp


def masked_otsu_threshold(img, mask):
    """
    Calculates threshold for Otsu binarisation of a masked region of an image.

    Based on Otsu binarisation example from:
    https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

    Parameters: 
    img: 8-bit single channel image
    mask: binary mask with same dimensions as img

    Returns: 
    int: Otsu threshold value
    """

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    masked = cv2.bitwise_and(blur, blur, mask=mask)  # mask image
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([masked], [0], None, [256], [0, 256])
    hist[0] = 0  # set black pixel count (from mask) to zero
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255]-Q[i]  # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    return thresh


def main():
    # setup config and logging
    config = configparser.ConfigParser()
    config.read('settings\\KITTI.ini')
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

    # get mask
    vp = get_vp(img, config)
    roi = get_roi(img, vp, config)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_otsu = masked_otsu_threshold(img, roi)
    otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print("custom={}, default={}".format(masked_otsu, otsu))


if __name__ == '__main__':
    main()

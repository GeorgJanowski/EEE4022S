# Detect road marking of cityscapes dataset.
# - Get ROI using semantic segmented image
# - Perform modified adaptive threshold on ROI
# - Perform global threshold on ROI
# - Bitwise and to get road markings

from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

from masked_adaptive_threshold import masked_adaptive_threshold
from masked_otsu_threshold import masked_otsu_threshold

def get_cityscapes_rm_da(img, semantics):
    """Applies two adaptive filters to find road markings"""

    semantics_road = (128,64,128)
    mask_road = cv2.inRange(semantics, semantics_road, semantics_road)

    # cv2.imwrite('temp/cityscapes_road_mask.png', mask_road)

    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height), (0, int(height * 3/4)), (width, int(height * 3/4)), (width, height)]])
    mask_bottom = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask_bottom, vertices, 255)
    mask_top = cv2.bitwise_not(mask_bottom)

    mask_road_top = cv2.bitwise_and(mask_road, mask_road, mask=mask_top)
    mask_road_bottom = cv2.bitwise_and(mask_road, mask_road, mask=mask_bottom)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.bitwise_and(img, img, mask=mask)
    thr_adaptive_top = masked_adaptive_threshold(img, mask_road_top, max_value=255, size=101, C=-15)
    thr_adaptive_top = cv2.convertScaleAbs(thr_adaptive_top)
    thr_adaptive_top = cv2.bitwise_and(thr_adaptive_top, thr_adaptive_top, mask=mask_road_top)
    thr_adaptive_bottom = masked_adaptive_threshold(img, mask_road_bottom, max_value=255, size=251, C=-15)
    thr_adaptive_bottom = cv2.convertScaleAbs(thr_adaptive_bottom)
    thr_adaptive_bottom = cv2.bitwise_and(thr_adaptive_bottom, thr_adaptive_bottom, mask=mask_road_bottom)
    thr_adaptive_combined = cv2.bitwise_or(thr_adaptive_top, thr_adaptive_bottom)

    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(3,1,2)
    plt.imshow(thr_adaptive_top, cmap='gray')
    plt.subplot(3,1,3)
    plt.imshow(thr_adaptive_bottom, cmap='gray')

    return thr_adaptive_combined

def get_cityscapes_rm(img, semantics, size, C):
    """Applies one adaptive filter to find road markings"""
    semantics_road = (128,64,128)
    mask = cv2.inRange(semantics, semantics_road, semantics_road)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_and(img, img, mask=mask)
    # thr_otsu = masked_otsu_threshold(img, mask)
    # _, thr_global = cv2.threshold(img, thr_otsu, 255, cv2.THRESH_BINARY)
    thr_adaptive = masked_adaptive_threshold(img, mask, max_value=255, size=size, C=C)
    # thr_adaptive = cv2.convertScaleAbs(thr_adaptive)
    # thr_combined = cv2.bitwise_and( thr_adaptive, thr_adaptive, mask=thr_global)
    thr_adaptive = cv2.bitwise_and(thr_adaptive, thr_adaptive, mask=mask)

    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.imshow(thr_global)
    # plt.subplot(3,1,2)
    # plt.imshow(thr_adaptive)
    # plt.subplot(3,1,3)
    # plt.imshow(thr_combined)

    return thr_adaptive

def main():
    img_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\leftImg8bit_trainvaltest\\leftImg8bit\\train\\aachen'
    img_filename = 'aachen_000132_000019_leftImg8bit.png'
    semantics_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\gtFine_trainvaltest\\gtFine\\train\\aachen'
    semantics_filename = 'aachen_000132_000019_gtFine_color.png'
    #132, 
    img = cv2.imread(img_dir + '\\' + img_filename)
    semantics = cv2.imread(semantics_dir + '\\' + semantics_filename)
    # aachen_000101_000019_leftImg8bit.png

    rm_small = get_cityscapes_rm(img, semantics, 51, C=-10)
    rm_large = get_cityscapes_rm(img, semantics, 301, C=-10)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rm_default = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -10)
    semantics_road = (128,64,128)
    mask = cv2.inRange(semantics, semantics_road, semantics_road)
    rm_default = cv2.bitwise_and(rm_default, rm_default, mask=mask)

    # rm = get_cityscapes_rm_da(img, semantics)
    # cv2.imwrite('temp/cityscapes_rm_mask.png', rm)
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(2, 1, 2)
    # plt.imshow(rm, cmap='gray')
    # plt.show()

    # cv2.imwrite('temp/adaptive_default.png', rm_default)
    # cv2.imwrite('temp/adaptive_large.png', rm_large)
    # cv2.imwrite('temp/adaptive_small.png', rm_small)

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.imshow(rm_default, cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(rm_small, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(rm_large, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
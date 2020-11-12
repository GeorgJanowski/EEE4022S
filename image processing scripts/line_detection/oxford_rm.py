# Detect road marking of oxford robotcar dataset in overcast conditions.
# - Get ROI using from mask image
# - Perform masked adaptive threshold on ROI

# notes
# overcast: masked_adaptive_threshold(img, mask, max_value=255, size=71, C=-30)
# night: masked_adaptive_threshold(img, mask, max_value=255, size=71, C=-20)


from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import configparser

from masked_adaptive_threshold import masked_adaptive_threshold
from masked_otsu_threshold import masked_otsu_threshold


def get_oxford_rm(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr_adaptive = masked_adaptive_threshold(
        img, mask, max_value=255, size=51, C=-20)
    thr_adaptive = cv2.convertScaleAbs(thr_adaptive)
    thr_adaptive = cv2.bitwise_and(thr_adaptive, thr_adaptive, mask=mask)
    kernel = np.ones((2, 2))
    morph = cv2.morphologyEx(thr_adaptive, cv2.MORPH_OPEN, kernel)
    dil = cv2.dilate(morph, kernel)

    return dil


def get_vp_mask(img, vp):
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height), (0, int(height * 2/3)),
                          (vp[0], vp[1]), (width, int(height * 2/3)), (width, height)]])
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def main():
    config = configparser.ConfigParser()
    config.read('settings\\oxford.ini')
    src_dir = config['data']['src_dir']
    img_filename = config['data']['file_name']
    mask_path = './settings/oxford_mask.png'
    img = cv2.imread(src_dir + '\\' + img_filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    rm = get_oxford_rm(img, mask)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(rm, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

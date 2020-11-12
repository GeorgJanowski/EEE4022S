# Perform adatptive thresholding on masked area of image.
# This should handle variations in illumination and deal with ROI edges.
# Example code from: https://stackoverflow.com/questions/9842127/using-a-mask-with-an-adaptive-threshold

from cv2 import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def thresh(a, b, max_value, C):
    return max_value if a > b - C else 0


v_thresh = np.vectorize(thresh)


def masked_adaptive_threshold(img, mask, max_value, size, C):
    '''Apply adatptive threshold ignoring masked regions'''
    kernel = np.ones((size, size), dtype='d')
    kernel[(size - 1) // 2, (size - 1) // 2] = 0
    conv = signal.fftconvolve(img, kernel, mode='same')
    num_neighbours = signal.fftconvolve(mask/255.0, kernel, mode='same')
    mean_conv = conv / num_neighbours
    return v_thresh(img, mean_conv, max_value, C)


def main():
    img_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\leftImg8bit_trainvaltest\\leftImg8bit\\train\\aachen'
    img_filename = 'aachen_000010_000019_leftImg8bit.png'
    semantics_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\gtFine_trainvaltest\\gtFine\\train\\aachen'
    semantics_filename = 'aachen_000010_000019_gtFine_color.png'
    img = cv2.imread(img_dir + '\\' + img_filename, cv2.IMREAD_GRAYSCALE)
    semantics = cv2.imread(semantics_dir + '\\' + semantics_filename)
    mask = cv2.inRange(semantics, (128, 64, 128), (128, 64, 128))

    img = cv2.bitwise_and(img, img, mask=mask)
    _, thr_global = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    thr_adaptive = masked_adaptive_threshold(
        img, mask, max_value=255, size=201, C=-15)
    thr_adaptive = cv2.convertScaleAbs(thr_adaptive)

    result = cv2.bitwise_and(thr_adaptive, thr_adaptive, mask=thr_global)

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('img')
    plt.subplot(2, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title('result')
    plt.subplot(2, 2, 3)
    plt.imshow(thr_adaptive, cmap='gray')
    plt.title('thr_adaptive')
    plt.subplot(2, 2, 4)
    plt.imshow(thr_global, cmap='gray')
    plt.title('thr_global')
    plt.show()


if __name__ == '__main__':
    main()

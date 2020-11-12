# Various measures of the difference between two images.

from cv2 import cv2
import math

from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

def ssim_wrapper(img_1, img_2):
    return structural_similarity(img_1, img_2, multichannel=True)


def l1_diff(img_1, img_2):
    """Calculate pixel wise difference between two colour images with l1 norm."""

    if (img_1.shape != img_2.shape):
        raise ValueError("Images must have same dimensions.")

    diff = 0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            for k in range(img_1.shape[2]):
                diff += abs(int(img_1[i][j][k]) - int(img_2[i][j][k]))

    return (diff / img_1.size) / 255


def l2_diff(img_1, img_2):
    """Calculate pixel wise difference between two colour images with l2 norm."""

    if (img_1.shape != img_2.shape):
        raise ValueError("Images must have same dimensions.")

    diff = 0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            for k in range(img_1.shape[2]):
                diff += (int(img_1[i][j][k]) - int(img_2[i][j][k])) ** 2

    return math.sqrt(diff / img_1.size) / 255


def edge_diff(img_1, img_2):
    """
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    """

    if (img_1.shape != img_2.shape):
        raise ValueError("Images must have same dimensions.")

    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    sobel_1 = cv2.Sobel(gray_1, cv2.CV_64F, 1, 0, None, ksize=5)
    sobel_2 = cv2.Sobel(gray_2, cv2.CV_64F, 1, 0, None, ksize=5)

    diff = 0
    for i in range(sobel_1.shape[0]):
        for j in range(sobel_1.shape[1]):
            diff += abs(sobel_1[i][j] - sobel_2[i][j])

    return (diff / sobel_1.size) / 255


def hist_diff(img_1, img_2):
    """ Calculates distance between images based on the distance between their
    histograms. Example found at:
    https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
    """

    hist_1 = cv2.calcHist([img_1], [0, 1, 2], None, [
                          8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_1 = cv2.normalize(hist_1, hist_1).flatten()
    hist_2 = cv2.calcHist([img_2], [0, 1, 2], None, [
                          8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_2 = cv2.normalize(hist_2, hist_2).flatten()

    return cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA)


def get_diff(img_1, img_2, diff_method):
    if diff_method == 'l1':
        return l1_diff(img_1, img_2)
    elif diff_method == 'l2':
        return l2_diff(img_1, img_2)
    elif diff_method == 'edge':
        return edge_diff(img_1, img_2)
    elif diff_method == 'hist':
        return hist_diff(img_1, img_2)
    elif diff_method == 'ssim':
        return ssim_wrapper(img_1, img_2)
    else:
        raise ValueError('unrecognised option \'' + diff_method + '\'.')


def main():
    src_dir = 'C:\\Users\\Georg\\OneDrive\\Documents\\EEE4022S\\code\\classic machine vision\\test_images\\'
    filename_1 = 'aachen_1_input.jpg'
    # filename_2 = 'aachen_1_classic.jpg'
    filename_3 = 'aachen_1_gan.jpg'

    img_1 = cv2.imread(src_dir + filename_1)
    # img_2 = cv2.imread(src_dir + filename_2)
    img_3 = cv2.imread(src_dir + filename_3)

    l1_11 = l1_diff(img_1, img_1)
    l1_13 = l1_diff(img_1, img_3)

    edge_11 = edge_diff(img_1, img_1)
    edge_13 = edge_diff(img_1, img_3)

    hist_11 = hist_diff(img_1, img_1)
    hist_13 = hist_diff(img_1, img_3)

    print('l1_11:', l1_11, ' l1_13:', l1_13)
    print('hist_11:', hist_11, ' hist_13:', hist_13)
    print('edge_11:', edge_11, ' edge_13:', edge_13)


if __name__ == '__main__':
    main()

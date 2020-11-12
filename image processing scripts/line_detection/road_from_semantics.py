# get binary mask of road from semantic segmentation image

from cv2 import cv2
from matplotlib import pyplot as plt

from pathlib import Path
from os import listdir, path


def binarise_semantics(img_semantic):
    color = (128, 64, 128)  # color of road in semantic image
    img_bin = cv2.inRange(img_semantic, color, color)
    return img_bin


def main():
    src_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\KITTI\\data_semantics\\training\\semantic_rgb'
    filename = '000000_10.png'
    img = cv2.imread(src_dir + '\\' + filename, cv2.IMREAD_UNCHANGED)

    img_bin = binarise_semantics(img)
    print(img_bin.shape, img_bin.dtype)

    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(img_bin, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()

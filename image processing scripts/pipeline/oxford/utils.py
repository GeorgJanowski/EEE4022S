# Contains some simple image formating to produce test images for GAN
# - resize images
# - produce double image

from cv2 import cv2
import numpy as np
from os import path
import glob


src_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\kitti\\test_256x256_kitti1\\'
dst_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\kitti\\test_512x256\\'
file_paths = glob.glob(src_dir + '*.jpg')

# nr = 1
# for file_path in file_paths:
#     filename = path.basename(file_path)[:-3] + 'jpg'
#     img = cv2.imread(file_path)
#     img_resized = cv2.resize(img, (256, 256))
#     cv2.imwrite(dst_dir + filename, img_resized)
#     print(nr)
#     nr += 1

nr = 1
for file_path in file_paths:
    filename = path.basename(file_path)
    img = cv2.imread(file_path)
    blank = np.zeros((256, 256, 3), np.uint8)
    img_double = np.concatenate((blank, img), axis=1)
    cv2.imwrite(dst_dir + filename, img_double)
    print(nr)
    nr += 1

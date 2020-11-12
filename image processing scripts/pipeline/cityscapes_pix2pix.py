# Creates images to be used in pix2pix GAN

from cityscapes_rm import get_cityscapes_rm_da
import degrade_rm
from os import listdir, path
from pathlib import Path
import matplotlib.pyplot as plt
from cv2 import cv2  # fixes linting bug
import numpy as np
import logging
import configparser
import glob
import sys
sys.path.insert(0, '.\\line_detection')
sys.path.insert(0, '.\\line_degradation')


# setup config and logging
config = configparser.ConfigParser()
config.read('settings\\cityscapes.ini')
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# dst_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\temp'
dst_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\temp'

image_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\leftImg8bit_trainvaltest\\leftImg8bit\\train'
# semantics_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\gtFine_trainvaltest\\gtFine\\train'
# rm_removed_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\leftImg8bit_rm_removed\\train'
rm_degraded_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\leftImg8bit_rm_degraded2'
# rm_parts_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\cityscapes\\leftImg8bit_rm_parts'

images = glob.glob(image_dir + '\\*\\*leftImg8bit.png')[:610]
# semantics = glob.glob(semantics_dir + '\\*\\*gtFine_color.png')
# rm_removed = glob.glob(rm_removed_dir + '\\*.jpg')[:610]
rm_degraded = glob.glob(rm_degraded_dir + '\\*.jpg')
# rm_parts = glob.glob(rm_parts_dir + '\\*.jpg')

counter = 1

# resize and combine images
for input_path, output_path in zip(images, rm_degraded):
    img_input = cv2.imread(input_path)
    img_output = cv2.imread(output_path)

    img_input = cv2.resize(img_input, (256, 256))
    img_output = cv2.resize(img_output, (256, 256))

    example = np.concatenate((img_output, img_input), axis=1)
    cv2.imwrite(dst_dir + '\\' + f"{counter:04d}" + '.jpg', example)

    print(counter)
    counter += 1

# remove road markings
# for img_path, sem_path in zip(images, semantics):
#     img_filename = path.basename(img_path)
#     img_input = cv2.imread(img_path)
#     sem = cv2.imread(sem_path)

#     mask = get_cityscapes_rm_da(img_input, sem)
#     img_output = degrade_rm.degrade_rm(img_input, mask, config)

#     # img_input = cv2.resize(img_input, (256,256))
#     # img_output = cv2.resize(img_output, (256,256))

#     # example = np.concatenate((img_output, img_input), axis=1)
#     # cv2.imwrite(dst_dir + '\\' + str(counter) + '.jpg', example)

#     print(dst_dir + '\\' + img_filename[:-3] + 'jpg')
#     cv2.imwrite(dst_dir + '\\' + img_filename[:-4] + '_rmp.jpg', img_output)

#     # plt.subplot(2,1,1)
#     # plt.imshow(img_input)
#     # plt.subplot(2,1,2)
#     # plt.imshow(img_output)
#     # plt.show()

# #     if counter == 610: break

# #     print(counter)
# #     counter += 1

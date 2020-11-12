# Creates a video using all the images in a folder.
# Adds labels to images.

from cv2 import cv2
import numpy as np
import glob

# get image paths
src_dir = 'C:\\Users\\Georg\\Desktop\\datasets\\results\\oxford_test_256x256_1000_2014-06-26-08-53-56\\'
files = glob.glob(src_dir + '*.jpg')
print(len(files))

# read images
img_array = []
for filename in files:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# add labels to images
# for i in range(len(img_array)):
#     cv2.putText(img_array[i], '[' + str(i) + '] ' + files[i], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# write video
out = cv2.VideoWriter('oxford_test_256x256_1000_2014-06-26-08-53-56.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

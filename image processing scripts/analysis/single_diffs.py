# Diffs between single input, target and output

from cv2 import cv2
from matplotlib import pyplot as plt

from img_diff import get_diff

diff_methods = ['l1', 'l2', 'edge', 'hist', 'ssim']
EXAMPLE_PATH = './test_images/68_example.jpg'
OUTPUT_PATH = './test_images/68_output.jpg'

example_img = cv2.imread(EXAMPLE_PATH)
input_img = example_img[:, 256:, :]
target_img = example_img[:, 0:256, :]
ouput_img = cv2.imread(OUTPUT_PATH)

diffs = {}
for diff_method in diff_methods:
    diffs[diff_method] = {}
    diffs[diff_method]['input_target'] = get_diff(input_img, target_img, diff_method)
    diffs[diff_method]['input_output'] = get_diff(input_img, ouput_img, diff_method)
    diffs[diff_method]['target_output'] = get_diff(target_img, ouput_img, diff_method)

print(diffs)

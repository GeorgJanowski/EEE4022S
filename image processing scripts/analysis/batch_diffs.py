# Compares target images in target_path to multiple output image in
# output_folders in output_path.

import glob
from matplotlib import pyplot as plt
from cv2 import cv2

import img_diff


MAX_IMAGES = 500 # smaller sample size for debugging

# other
# target_path = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\cityscapes\\256x256_610_rmr1\\test\\'
# output_path = 'C:\\Users\\Georg\\Desktop\\datasets\\results\\dataset_generalisation\\'
# output_folders = ['city_city']
# diff_methods = ['l1', 'l2', 'edge', 'hist', 'ssim']

# input benchmark, compare target output to input
# target_path = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\cityscapes\\256x256_610_rmr1\\test\\'
# output_folders = ['inputs']
# diff_methods = ['l1', 'l2', 'edge', 'hist', 'ssim']

# lambda
# target_path = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\cityscapes\\256x256_610_rmr1\\test\\'
# output_path = 'C:\\Users\\Georg\\Desktop\\datasets\\results\\lambda_experiment\\'
# output_folders = ['l100', 'l200', 'l400', 'l1000', 'l2000']
# output_folders = ['l4000', 'l10000']
# diff_methods = ['l1', 'l2', 'edge', 'hist', 'ssim']

# batch_size
# target_path = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\cityscapes\\256x256_610_rmr1\\test\\'
# output_path = 'C:\\Users\\Georg\\Desktop\\datasets\\results\\batch_experiment\\'
# output_folders = ['b1', 'b2', 'b5', 'b10']
# diff_methods = ['l1', 'l2', 'edge', 'hist', 'ssim']

# mirror
# target_path = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\cityscapes\\256x256_610_rmr1\\test\\'
# output_path = 'C:\\Users\\Georg\\Desktop\\datasets\\results\\mirror_experiment\\'
# output_folders = ['l100_mirror', 'l100_no_mirror', 'l400_mirror', 'l400_no_mirror']
# diff_methods = ['l1', 'l2', 'edge', 'hist', 'ssim']

# l2 loss
# target_path = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\cityscapes\\256x256_610_rmr1\\test\\'
# output_path = 'C:\\Users\\Georg\\Desktop\\datasets\\results\\l2_loss\\'
# output_folders = ['cityscapes_rmr_610_100_l100_L2', 'cityscapes_rmr_610_100_l2000_L2']
# diff_methods = ['l1', 'l2', 'edge', 'hist', 'ssim']

# size
target_path = 'C:\\Users\\Georg\\Desktop\\datasets\\pix2pix\\cityscapes\\512x512_610_rmr1\\test\\'
output_path = 'C:\\Users\\Georg\\Desktop\\datasets\\results\\size_experiment\\'
output_folders = ['256_256', '256_512', '512_512']
diff_methods = ['l1', 'l2', 'edge', 'hist', 'ssim']

# load target images
example_filenames = glob.glob(target_path + '*.jpg')
input_images = []
target_images = []
for example_filename in example_filenames:
    img = cv2.imread(example_filename)
    input_images.append(img[:, 512:, :])
    target_images.append(img[:, 0:512, :])

# load output images
output_images = {}
for output_folder in output_folders:
    counter = 0
    file_paths = glob.glob(output_path + output_folder + '\\*.jpg')
    output_images[output_folder] = []
    for file_path in file_paths:
        output_images[output_folder].append(cv2.imread(file_path))
        if counter == MAX_IMAGES:
            break
        counter += 1

# target_images = input_images
# output_images[output_folders[0]] = input_images

# check images are correct
img_num = 12
plt.figure()
plt.subplot(2,1,1)
plt.imshow(target_images[img_num])
plt.subplot(2,1,2)
plt.imshow(output_images[output_folders[0]][img_num])
plt.show()

# calculate diffs
diffs = {}
for diff_method in diff_methods:
    diffs[diff_method] = {}
    for folder_name, folder_images in output_images.items():
        diffs[diff_method][folder_name] = []
        for target_image, output_image in zip(target_images, folder_images):
            diffs[diff_method][folder_name].append(
                img_diff.get_diff(target_image, output_image, diff_method))

# calculate average each diff
avg_diffs = {}
for diff_key, diff_values in diffs.items():
    avg_diffs[diff_key] = {}
    for model_key, values in diff_values.items():
        avg_diffs[diff_key][model_key] = sum(values) / len(values)

print(avg_diffs)

# plot results
for diff_key, diff_values in diffs.items():
    fig = plt.figure()
    fig.suptitle(diff_key)
    for model_key, values in diff_values.items():
        plt.plot(values, label=model_key)
    plt.legend()
plt.show()

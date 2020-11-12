# Some functions to add Perlin noise to images.

from road_marking_detection import get_rm
from road_area_detection import get_roi
from vanishing_point_detection import get_vp
import noise
import numpy as np
from cv2 import cv2  # fixes linting bug

import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir, path

import sys
sys.path.insert(0, '.\\line_detection')


def generate_pnoise(shape):
    # Perlin noise perameters
    scale = 100.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=0)

    world = (world + 0.5) * 255
    world = cv2.convertScaleAbs(world)

    return world


def projective_transformation(img, vp):
    rows, cols = img.shape[:2]
    src_points = np.float32(
        [[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    # dst_points = np.float32([[int(0.33*cols),int(rows/2)], [int(0.33*cols) + 100,int(rows/2)], [0,rows-1], [cols-1,rows-1]])
    dst_points = np.float32(
        [[vp[0] - 100, vp[1]], [vp[0] + 100, vp[1]], [-cols, 2*rows], [2*cols, 2*rows]])

    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows))

    return img_output


def overlay_images(img1, img2, mask):
    # img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img1, img1, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)

    return dst


def add_pnoise(img, noise, mask):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(img.shape, noise.shape, mask.shape)
    print(img.dtype, noise.dtype, mask.dtype)

    masked_noise = cv2.bitwise_and(noise, noise, mask=mask)
    # _, masked_noise = cv2.threshold(masked_noise, 200, 255, cv2.THRESH_BINARY)
    masked_noise = cv2.cvtColor(masked_noise, cv2.COLOR_GRAY2RGB)

    # alpha = 0.5
    # beta = ( 1.0 - alpha )
    # res = cv2.addWeighted(img, alpha, masked_noise, beta, 0.0)

    # res = cv2.bitwise_and(img, masked_noise)

    res = img - masked_noise

    return res


def main():
    # load image
    source_folder = Path(
        'C:\\Users\\Georg\\Desktop\\datasets\\KITTI\\2011_09_26_drive_0056_extract\\2011_09_26\\2011_09_26_drive_0056_extract\\image_02\\data')
    # dest_folder = Path("dest_data")
    files = [f for f in listdir(source_folder) if path.isfile(
        path.join(source_folder, f))]
    img = cv2.imread(str(source_folder/files[100]), cv2.IMREAD_UNCHANGED)

    # get mask of road markings
    vp = get_vp(img)
    roi = get_roi(img, vp)
    mask = get_rm(img, roi)

    # get noise
    pnoise = generate_pnoise(img.shape[:2])
    print(pnoise.dtype)

    # projective transform
    # pnoise_proj = projective_transformation(pnoise,vp)

    # add noise
    # img_output = add_pnoise(img, pnoise_proj, mask)
    pnoise_color = cv2.cvtColor(pnoise, cv2.COLOR_GRAY2RGB)
    img_output = overlay_images(img, pnoise_color, mask)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('img')
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.title('mask')
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pnoise')
    plt.imshow(pnoise, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title('img_output')
    plt.imshow(img_output, cmap='gray')

    plt.figure()
    plt.imshow(img_output)
    plt.show()


if __name__ == '__main__':
    main()

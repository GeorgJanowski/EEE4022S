# Estimate where the road is by estimating the vanishing point from lane line
# markings.

import numpy as np
from cv2 import cv2  # fix linter bug
import math

import configparser
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir, path


def get_line_from_coordinates(coordinates):
    """ 
    Returns gradient and y-intercept of a line given two points on the line.
    """
    x1, y1, x2, y2 = coordinates
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return (m, c)


def calculate_intercept_from_pnts(line1, line2):
    """ 
    Returns the coordinates of the intercept of two lines defined by two points.
    """
    # get equation of two lines from coordinates
    m1, c1 = get_line_from_coordinates(line1)
    m2, c2 = get_line_from_coordinates(line2)
    # calculate intercept
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return (int(x), int(y))


def calculate_intercept_from_eqs(eq1, eq2):
    """
     Returns the coordinates of the intercept of two lines defined as y=mx+c.
    """
    # get equation of two lines from coordinates
    m1, c1 = eq1
    m2, c2 = eq2
    # calculate intercept
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    if (math.isfinite(x) or math.isfinite(y)):
        return (int(x), int(y))
    return (0, 0)


def get_vp(img_input, config):
    """ 
    Return coordinates of vanishing point calculated as the intersection
    of the two lines with min and max gradients.
    If no suitable vanishing point is found a default will be returned.
    """

    # set default vp TODO: get better estimate
    vp = (config['vp'].getint('default_x'), config['vp'].getint('default_y'))

    # get edges
    img = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)
    kernel_size = config['vp'].getint('kernel_size')
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img = cv2.Canny(img, config['vp'].getint(
        'low_thr'), config['vp'].getint('high_thr'))

    # mask out upper half of image
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array(
        [[(0, height), (width, height), (width//2, height//2)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    img = cv2.bitwise_and(img, mask)

    # Hough transform to find lines
    hough = cv2.HoughLinesP(img, 2, np.pi / 180, 100, 0,
                            minLineLength=200, maxLineGap=50)
    if (hough is not None):
        # convert [x1,y1,x2,y2] to y=mx+c (m,c)
        hough_mc = [get_line_from_coordinates(line[0]) for line in hough]
        # remove lines that are unrealistically horizontal
        hough_mc = [line for line in hough_mc if (abs(line[0]) > 0.2)]

        # we need at least two lines to find vp
        if (len(hough_mc) > 1):
            # convert gradients to radiens
            hough_radc = [(math.atan(line[0]), line[1]) for line in hough_mc]
            # find index of lines with min and max gradients
            min_index = hough_radc.index(min(hough_radc))
            max_index = hough_radc.index(max(hough_radc))
            # find vp as intersection of two lines
            p_vp = calculate_intercept_from_eqs(
                hough_mc[min_index], hough_mc[max_index])

            # set vp to intersection if it is reasonable
            y_min = config['vp'].getint('y_min')
            y_max = config['vp'].getint('y_max')
            if p_vp[1] > y_min and p_vp[1] < y_max and p_vp[0] > 0 and p_vp[0] < width:
                vp = p_vp
            else:
                logging.info('Could not find vp.')

    # DEBUG: draw lines
    # x1, y1, x2, y2 = hough[5][0]
    # cv2.line(img_input, (x1, y1), (x2, y2), (0,255,0), 5)
    # x1, y1, x2, y2 = hough[9][0]
    # cv2.line(img_input, (x1, y1), (x2, y2), (0,255,0), 5)
    # for line in hough:
    #     x1, y1, x2, y2 = line[0]
    #     print(line[0])
    #     # Draws lines between two coordinates with green color and 5 thickness
    #     cv2.line(img_input, (x1, y1), (x2, y2), (0,255,0), 5)

    return vp


def main():
    # setup config and logging
    config = configparser.ConfigParser()
    config.read('settings\\kitti.ini')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # load image
    src_dir = Path(config['data']['src_dir'])
    if config['data'].getboolean('use_num'):
        files = [f for f in listdir(
            src_dir) if path.isfile(path.join(src_dir, f))]
        img_path = str(src_dir/files[config['data'].getint('file_num')])
    else:
        img_path = str(src_dir/config['data']['file_name'])
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # get vp
    vp = get_vp(img, config)

    # draw horizom line
    cv2.line(img, (0, vp[1]), (img.shape[1], vp[1]), (0, 0, 255), 2)

    # draw vanishing point
    cv2.line(img, vp, vp, (255, 0, 0), 12)

    # display image
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()

    # cv2.imwrite('temp/kitti_vp_detection.png', img)


if __name__ == '__main__':
    main()

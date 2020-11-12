# Generate 2D Perlin Noise with random offset

import noise
import numpy as np
from cv2 import cv2


def perlin_noise_2d(shape, scale=100.0, octaves=1, persistence=0.5, lacunarity=2.0):
    base = np.random.randint(0, shape[1])
    base = 0
    print('base:', base)

    img_noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            img_noise[i][j] = noise.pnoise2(i/scale,
                                            j/scale,
                                            octaves=octaves,
                                            persistence=persistence,
                                            lacunarity=lacunarity,
                                            repeatx=shape[1],
                                            repeaty=shape[0],
                                            base=base)

    return ((img_noise + 0.5) * 255).astype(np.uint8)


def main():
    shape = (1024, 1024)
    scale = 100.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    img = perlin_noise_2d(shape, scale, octaves, persistence, lacunarity)
    print(img.shape, img.dtype)
    print('min:', np.amin(img), '\tmax:', np.amax(img))

    cv2.imshow('img_noise', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

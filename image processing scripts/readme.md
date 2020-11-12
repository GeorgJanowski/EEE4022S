Code used to create training examples of degraded road markings.

- analysis: image comparison functions to compare output and target images.
- rm_degradation: degrade detected road markings using inpainting and noise.
- rm_detection: detect road markings by finding a region of interest for the road surface and then using thresholds.
- pipeline: detect and degrade batches of images.
- settings: contains parameter values and other details to be used with different datasets.

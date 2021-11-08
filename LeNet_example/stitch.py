import numpy as np
import argparse
import cv2

default_str = 'output/bayesian_predictions/trial_{}.png'
n = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", type=str, required=True, help="path to save output image")
    args = parser.parse_args()

    images = []
    for i in range(n):
        images.append(cv2.imread(default_str.format(i+1)))

    row1 = np.concatenate(images[:n//2], axis=1)
    row2 = np.concatenate(images[n//2:], axis=1)
    stitched_image = np.concatenate([row1, row2], axis=0)

    cv2.imwrite(args.plot, stitched_image)

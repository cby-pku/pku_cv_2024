'''
This is the main file for the project 2's Second method PDE
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d, pad
import os




def pde(img, loc, beta):
    ''' 
    The function to perform the pde update for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. beta: learning rate
    Return:
        img: the updated image
    '''

    # TODO

    return img


def main():
    # read the distorted image and mask image
    name = "sce"
    size = "big"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)



    beta = 1
    img_height, img_width, _ = distort.shape

    sweep = 100
    for s in tqdm(range(sweep)):
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                # TODO

        # TODO

        if s % 10 == 0:
            save_path = f"./result/{name}/{size}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(f"{save_path}/pde_{s}.bmp", distort)



if __name__ == "__main__":
    main()







        


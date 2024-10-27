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
import matplotlib.pyplot as plt
import json


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
    i, j = loc
    height, width, _ = img.shape
    
    neighbors = []
    if i > 0:
        neighbors.append(img[i-1,j,2])
    if i < height - 1:
        neighbors.append(img[i+1,j,2])
    if j > 0:
        neighbors.append(img[i,j-1,2])
    if j < width - 1:
        neighbors.append(img[i,j+1,2])
    
    avg_neighbor = np.mean(neighbors)

    img[i,j,2] = img[i,j,2] + beta * (avg_neighbor - img[i,j,2])

    return img


def main(name,size):
    # read the distorted image and mask image

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)



    beta = 0.1
    img_height, img_width, _ = distort.shape
    errors = []
    sweep = 100
    for s in tqdm(range(sweep)):
        for i in tqdm(range(img_height)):
            for j in range(img_width):
                # only change the channel red
                # TODO
                if mask[i,j,2] == 255:
                    distort = pde(distort, (i,j), beta)

        # TODO
        error = np.sum((distort[:,:,2] - ori[:,:,2])**2) / (img_height * img_width)
        errors.append(error)
        
        beta = min(1.0, beta + 0.01)
        
        if s % 10 == 0:
            save_path = f"./result_beta_annealing/{name}/{size}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(f"{save_path}/pde_{s}.bmp", distort)
        # Save errors to a JSON file
    save_path = f"./result/{name}/{size}"
    with open(f"{save_path}/errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=4, ensure_ascii=False)

    # Plot and save the error graph
    plt.figure()
    plt.plot(range(sweep), errors, label='Per Pixel Error', color='blue')
    plt.xlabel('Number of Sweeps')
    plt.ylabel('Error')
    plt.title(f'Error over Sweeps for {name} {size}')
    plt.legend()
    plt.savefig(f"{save_path}/error_plot.png")
    plt.close()



if __name__ == "__main__":
    name_list = ["sce", "room", "stone"]
    size_list = ["small", "big"]
    for name in name_list:
        for size in size_list:
            main(name, size)







        


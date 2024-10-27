'''
This is the main file for the project 2's first method Gibss Sampler
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
import os
from torch.nn.functional import conv2d, pad
import matplotlib.pyplot as plt
import json

color_red = '#FF6B6B'  # red
color_green = '#6BCB77'  # green
color_blue = '#4D96FF'  # blue
color_yellow = '#FFD93D'  # yellow

def cal_pot(gradient, norm):
    ''' 
    The function to calculate the potential energy based on the selected norm
    Parameters:
        gradient: the gradient of the image, can be nabla_x or nabla_y, numpy array of size:(img_height,img_width, )
        norm: L1 or L2
    Return:
        A term of the potential energy
    '''
    if norm == "L1":
        return abs(gradient)
    elif norm == "L2":
        return gradient**2 
    else:
        raise ValueError("The norm is not supported!")




def gibbs_sampler(img, loc, energy, beta, norm):
    ''' 
    The function to perform the gibbs sampler for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. energy: a scale, refers to the negative exponent
        4. beta: 1/(annealing temperature)
        5. norm: L1 or L2
    Return:
        img: the updated image
    '''
    
    energy_list = np.zeros((256,1))
    # get the size of the image
    img_height, img_width = img.shape

    # original pixel value
    original_pixel = img[loc[0], loc[1]]
    # TODO: calculate the energy
    # NOTE should only change the pixel value that are masked 
    # NOTE in the main function, it has already set the pixel value that are not masked to 255
    # NOTE Optmization, use the neighbors to calculate the energy instead of the whole image by convolution
    for pixel_value in range(256):
        img[loc[0], loc[1]] = pixel_value
        neighbors = []
        if loc[0] > 0:
            neighbors.append(img[loc[0] - 1, loc[1]])
        if loc[0] < img_height - 1:
            neighbors.append(img[loc[0] + 1, loc[1]])
        if loc[1] > 0:
            neighbors.append(img[loc[0], loc[1] - 1])
        if loc[1] < img_width - 1:
            neighbors.append(img[loc[0], loc[1] + 1])
        
        # calculate the energy
        energy_list[pixel_value] = sum(cal_pot(pixel_value - neighbor, norm) for neighbor in neighbors) - energy
        

    # normalize the energy
    # TODO NOTE Importantly! if the effect is not good, try to normalize the energy in a better way or use the energy_list directly
    energy_list = energy_list - energy_list.min()
    # energy_list = energy_list / energy_list.sum() # NOTE partial function 1/Z


    # calculate the conditional probability
    probs = np.exp(-energy_list * beta)
    # normalize the probs
    probs = probs / probs.sum()

    try:
        # inverse_cdf and updating the img
        # NOTE use the flatten() to make the probs a 1D array, (256,1) -> (256,) or (256,) -> (256,)
        new_pixel_value = np.random.choice(range(256), p=probs.flatten())
        img[loc[0], loc[1]] = new_pixel_value
    except:
        raise ValueError(f'probs = {probs}')
    return img

def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape, can be [[-1,1]] or [[1],[-1]] or [[1,-1]] or [[-1],[1]] ....
    Return:
        filtered_image: numpy array of shape (H, W)
    '''
    H, W = image.shape
    filter_H, filter_W = filter.shape
    filtered_image = np.zeros((H, W))
    
    # NOTE using the periodic boundary condition

    padded_image = np.pad(image, ((filter_H//2, filter_H//2), (filter_W//2, filter_W//2)), mode='wrap') 
    
    for i in range(H):
        for j in range(W):
            filtered_image[i,j] = np.sum(padded_image[i:i+filter_H, j:j+filter_W] * filter)

    return filtered_image

def main(name,size):
    # read the distorted image and mask image
    # name = "sce" # TODO need to change this methods 
    # size = "small"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)

    # calculate initial energy
    red_channel = distort[:,:,2]
    energy = 0

    #calculate nabla_x
    filtered_img = conv(red_channel, np.array([[-1,1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))

    # calculate nabla_y
    filtered_img = conv(red_channel, np.array([[-1],[1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))



    norm = "L2"
    beta = 0.1
    img_height, img_width, _ = distort.shape

    sweep = 100
    errors = []
    for s in tqdm(range(sweep)):
        # NOTE visiting all distorted pixels once is 1-sweep 
        for i in tqdm(range(img_height)):
            for j in range(img_width):
                # only change the channel red
                if mask[i,j,2] == 255:
                    distort[:,:,2] = gibbs_sampler(distort[:,:,2], [i,j], energy, beta, norm)
        # TODO
        # NOTE every sweep to implement annealing strategy
        beta = min(1.0, beta + 0.01)
        error = np.sum((distort[:,:,2] - ori[:,:,2])**2) / (img_height * img_width)
        errors.append(error)

        save_path = f"./result/{name}/{norm}/{size}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)
    # TODO restore the errors in the save_path as json
    with open(f"{save_path}/errors.json", "w",encoding="utf-8") as f:
        json.dump(errors, f,indent=4,ensure_ascii=False)
        
    plt.figure()
    plt.plot(range(sweep), errors, label='Per Pixel Error',color=color_blue)
    plt.xlabel('Number of Sweeps')
    plt.ylabel('Error')
    plt.title(f'Error over Sweeps for {name} {size}')
    plt.legend()
    plt.savefig(f"{save_path}/error_plot.png")
    plt.close()



if __name__ == "__main__":
    name_list = ["sce","room","stone"]
    size_list = ["small","big"]
    for name in name_list:
        for size in size_list:
            main(name,size)







        


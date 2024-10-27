import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
import os
from torch.nn.functional import conv2d, pad
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F

def cal_pot(gradient, norm):
    ''' 
    Calculates potential energy based on the chosen norm (L1 or L2).
    '''
    if norm == "L1":
        return torch.abs(gradient)
    elif norm == "L2":
        return gradient ** 2
    else:
        raise ValueError("The norm is not supported!")
    
def gibbs_sampler(img, loc, energy, beta, norm):
    energy_list = torch.zeros((256, 1), device='cuda')
    img_height, img_width = img.shape

    original_pixel = img[loc[0], loc[1]]
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
        
        energy_list[pixel_value] = sum(cal_pot(pixel_value - neighbor, norm) for neighbor in neighbors) - energy

    energy_list = energy_list - energy_list.min()
    probs = torch.exp(-energy_list * beta)
    probs = probs / probs.sum()

    try:
        new_pixel_value = torch.multinomial(probs.flatten(), 1).item()
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

def main(name, size):
    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"

    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)

    red_channel = torch.tensor(distort[:,:,2], device='cuda')
    energy = 0

    red_channel_cuda = red_channel.to('cuda')

    filtered_img = conv(red_channel_cuda.cpu().numpy(), np.array([[-1,1]]).astype(np.float64))
    energy += torch.sum(torch.abs(torch.tensor(filtered_img, device='cuda')), dim=(0,1))

    filtered_img = conv(red_channel_cuda.cpu().numpy(), np.array([[-1],[1]]).astype(np.float64))
    energy += torch.sum(torch.abs(torch.tensor(filtered_img, device='cuda')), dim=(0,1))

    norm = "L2"
    beta = 0.1
    img_height, img_width, _ = distort.shape

    sweep = 100
    errors = []
    for s in tqdm(range(sweep)):
        for i in tqdm(range(img_height)):
            for j in range(img_width):
                if mask[i,j,2] == 255:
                    distort[:,:,2] = gibbs_sampler(torch.tensor(distort[:,:,2], device='cuda'), [i,j], energy, beta, norm).cpu().numpy()
        beta = min(1.0, beta + 0.01)
        error = np.sum((distort[:,:,2] - ori[:,:,2])**2) / (img_height * img_width)
        errors.append(error)

        save_path = f"./result_acc/{name}/{norm}/{size}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)
    with open(f"{save_path}/errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=4, ensure_ascii=False)

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
'''
This is the code for project 1 question 2
Question 2: Verify the 1/f power law observation in natural images in Set A
'''
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
path = "./image_set/setA/"
color_red = '#FF6B6B'  # red
color_green = '#6BCB77'  # green
color_blue = '#4D96FF'  # blue
color_yellow = '#FFD93D'  # yellow

colorlist = [color_red, color_blue, 'black', color_green]
linetype = ['-', '-', '-', '-']
labellist = ["natural_scene_1.jpg", "natural_scene_2.jpg",
                 "natural_scene_3.jpg", "natural_scene_4.jpg"]

img_list = [cv2.imread(os.path.join(path,labellist[i]), cv2.IMREAD_GRAYSCALE) for i in range(4)]
def fft(img):
    ''' 
    Conduct FFT to the image and move the dc component to the center of the spectrum
    Tips: dc component is the one without frequency. Google it!
    Parameters:
        1. img: the original image
    Return:
        1. fshift: image after fft and dc shift
    '''
    fshift = img # Need to be changed
    # TODO: Add your code here
    
    f = np.fft.fft2(img) # 2D FFT
    fshift = np.fft.fftshift(f)

    return fshift

def amplitude(fshift):
    '''
    Parameters:
        1. fshift: image after fft and dc shift
    Return:
        1. A: the amplitude of each complex number
    '''

    # A = fshift # Need to be changed
    # TODO: Add your code here
    
    A = np.abs(fshift)

    return A

def xy2r(x, y, centerx, centery):
    ''' 
    change the x,y coordinate to r coordinate
    '''
    rho = math.sqrt((x - centerx)**2 + (y - centery)**2)
    return rho

def cart2porl(A,img):
    ''' 
    Finish question 1, calculate the A(f) 
    Parameters: 
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. f: the frequency list 
        2. A_f: the amplitude of each frequency
    Tips: 
        1. Use the function xy2r to get the r coordinate!
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)
    
    # TODO: Add your code here
    max_r = min(centerx, centery)
    f = np.arange(0, max_r + 1)
    
    A_f = np.zeros(len(f))
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            r = int(xy2r(x, y, centerx, centery))
            if r < len(f):
                A_f[r] += A[x, y]

    epsilon = 1e-6  
    A_f[1:] = A_f[1:] / (2 * np.pi * f[1:] + epsilon) 
    
    return f, A_f


def get_S_f0(A,img):
    ''' 
    Parameters:
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. S_f0: the S(f0) list
        2. f0: frequency list
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)

    # TODO: Add your code here
    
    basic_f = 1
    max_r = min(centerx,centery)
    # the frequency coordinate
    f0 = np.arange(0,max_r + 1,basic_f)
    
    S_f0 = np.zeros(len(f0))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            r = int(xy2r(x, y, centerx, centery))
            if r < len(f0):
                S_f0[r] += A[x, y] ** 2  
    
    epsilon = 1e-6
    S_f0 = S_f0 / (2 * np.pi * f0 + epsilon)  

    return S_f0, f0
    
def main():
    plt.figure(1)
    # q1

    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        f, A_f = cart2porl(A,img_list[i])
        plt.plot(np.log(f[1:190]),np.log(A_f[1:190]), color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("1/f law")
    plt.savefig("./pro2_result/f1_law.jpg", bbox_inches='tight', pad_inches=0.0,dpi=300)

    # q2
    plt.figure(2)
    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        S_f0, f0 = get_S_f0(A,img_list[i])
        plt.plot(f0[10:],S_f0[10:], color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("S(f0)")
    plt.savefig("./pro2_result/S_f0.jpg", bbox_inches='tight', pad_inches=0.0,dpi=300)
if __name__ == '__main__':
    main()

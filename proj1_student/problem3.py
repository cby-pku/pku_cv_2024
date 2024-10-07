'''
-----------------------------------------------
This is the code for project 1 question 3
A 2D scale invariant world
'''
import numpy as np
import cv2
r_min = 1
def inverse_cdf(x):
    ''' 
    Parameters:
        1. x: the random number sampled from uniform distribution
    Return:
        1. y: the random number sampled from the cubic law power
    '''
    # TODO: Add your code here
    y = 1 / np.sqrt(1 - x)
    return y
def GenLength(N):
    ''' 
    Function for generating the length of the line
    Parameters:
        1. N: the number of lines
    Return:
        1. random_length: N*1 array, the length of the line, sampled from sample_r
    Tips:
        1. Using inverse transform sampling. Google it!
    '''
    # sample a random number from uniform distribution
    U = np.random.random(N)
    random_length = inverse_cdf(U)
    return random_length

def DrawLine(points,rad,length,pixel,N):
    ''' 
    Function for drawing lines on a image
    Parameters:
        1. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        2. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        3. length: N*1 array, the length of the line, sampled from sample_r
        4. pixel: the size of the image
        5. N: the number of lines
    Return:
        1. bg: the image with lines
    '''
    # background
    bg = 255*np.ones((pixel,pixel)).astype('uint8')

    # TODO: Add your code here
    
    for i in range(N):
        x_s, y_s = points[i]
        x_e = x_s + length[i] * np.cos(rad[i])
        y_e = y_s + length[i] * np.sin(rad[i])
        
        x_e = np.clip(x_e, 0, pixel - 1)
        y_e = np.clip(y_e, 0, pixel - 1)
        
        cv2.line(bg, (int(x_s), int(y_s)), (int(x_e), int(y_e)), color=(0, 0, 0), thickness=1)

    cv2.imwrite('./pro3_result/'+str(pixel)+'.png', bg)
    return bg

def solve_q1(N = 5000,pixel = 1024):
    ''' 
    Code for solving question 1
    Parameters:
        1. N: the number of lines
        2. pixel: the size of the image
    '''
    # Generating length
    length = GenLength(N)

    points = np.random.uniform(0, pixel, size=(N, 2))
    

    rad = np.random.uniform(0, 2 * np.pi, size=N)
    
    # Discard lines shorter than 1 pixel
    valid_indices = length >= 1
    points = points[valid_indices]
    rad = rad[valid_indices]
    length = length[valid_indices]

    image = DrawLine(points, rad, length, pixel, len(length))
    
    return image, points, rad, length

def DownSampling(img,points,rad,length,pixel,N,rate):
    ''' 
    Function for down sampling the image
    Parameters:
        1. img: the image with lines
        2. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        3. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        4. length: N*1 array, the length of the line
        5. pixel: the size of the image
        6. rate: the rate of down sampling
    Return:
        1. image: the down sampled image
    Tips:
        1. You can use Drawline for drawing lines after downsampling the components
    '''
  
    # TODO: Add your code here
    # Downsample the line properties
    points_downsampled = points / rate
    length_downsampled = length / rate
    
    # Remove lines shorter than 1 pixel
    valid_indices = length_downsampled >= 1
    points_downsampled = points_downsampled[valid_indices]
    length_downsampled = length_downsampled[valid_indices]
    rad_downsampled = rad[valid_indices]
    N_downsampled = len(length_downsampled)  
    
    new_pixel_size = int(pixel / rate)
    img_downsampled = DrawLine(points_downsampled, rad_downsampled, length_downsampled, new_pixel_size, N_downsampled)
    
    return img_downsampled

def crop(image1,image2,image3):
    ''' 
    Function for cropping the image
    Parameters:
        1. image1, image2, image3: I1, I2, I3
    '''
    
    # TODO: Add your code here
    def random_crop(img, crop_size=128):
        h, w = img.shape
        top = np.random.randint(0, h - crop_size)
        left = np.random.randint(0, w - crop_size)
        return img[top:top+crop_size, left:left+crop_size]
    
    crop1 = random_crop(image1)
    crop2 = random_crop(image2)
    crop3 = random_crop(image3)
    
    cv2.imwrite('./pro3_result/crop/crop_1024.png', crop1)
    cv2.imwrite('./pro3_result/crop/crop_512.png', crop2)
    cv2.imwrite('./pro3_result/crop/crop_256.png', crop3)
    return


import matplotlib.pyplot as plt

def double_crop(image1, image2, image3):
    ''' 
    Function for cropping 128x128 patches from the images.
    Parameters:
        1. image1, image2, image3: I1 (1024x1024), I2 (512x512), I3 (256x256)
    '''
    
    def random_crop(img, crop_size=128):
        '''Randomly crops a 128x128 patch from the image.'''
        h, w = img.shape
        top = np.random.randint(0, h - crop_size)
        left = np.random.randint(0, w - crop_size)
        return img[top:top+crop_size, left:left+crop_size]
    
    # Crop 2 patches from each image
    crop_1_1 = random_crop(image1)
    crop_1_2 = random_crop(image1)
    
    crop_2_1 = random_crop(image2)
    crop_2_2 = random_crop(image2)
    
    crop_3_1 = random_crop(image3)
    crop_3_2 = random_crop(image3)
    
    # Save the crops
    cv2.imwrite('./pro3_result/crop/crop_1024_1.png', crop_1_1)
    cv2.imwrite('./pro3_result/crop/crop_1024_2.png', crop_1_2)
    cv2.imwrite('./pro3_result/crop/crop_512_1.png', crop_2_1)
    cv2.imwrite('./pro3_result/crop/crop_512_2.png', crop_2_2)
    cv2.imwrite('./pro3_result/crop/crop_256_1.png', crop_3_1)
    cv2.imwrite('./pro3_result/crop/crop_256_2.png', crop_3_2)
    
    # Return all the crops
    return [crop_1_1, crop_1_2, crop_2_1, crop_2_2, crop_3_1, crop_3_2]


def main_report():
    N = 10000
    pixel = 1024
    

    image_1024, points, rad, length = solve_q1(N, pixel)
    

    image_512 = DownSampling(image_1024, points, rad, length, pixel, N, rate=2)
    image_256 = DownSampling(image_1024, points, rad, length, pixel, N, rate=4)
    

    cropped_patches = double_crop(image_1024, image_512, image_256)
    



def main():
    N = 10000
    pixel = 1024
    image_1024, points, rad, length = solve_q1(N,pixel)
    # 512 * 512
    image_512 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 2)
    # 256 * 256
    image_256 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 4)
    crop(image_1024,image_512,image_256)
    
if __name__ == '__main__':
    # main()
    main_report()

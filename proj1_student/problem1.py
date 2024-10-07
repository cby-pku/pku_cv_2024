'''
This is the code for project 1 question 1
Question 1: High kurtosis and scale invariance
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gennorm, fit, norm
from scipy.optimize import curve_fit
import scipy.special
from scipy.special import gamma
from tqdm import tqdm
from math import sqrt


color_red = '#FF6B6B'  # red
color_green = '#6BCB77'  # green
color_blue = '#4D96FF'  # blue
color_yellow = '#FFD93D'  # yellow


data_repo = "./image_set"
set_repo = ['setA','setB','setC']
img_name_list = []
def read_img_list(set):
    '''
    Read images from the corresponding image set
    '''
    global img_name_list
    img_list = os.listdir(os.path.join(data_repo,set))
    img_list.sort()
    img_name_list.append(img_list)
    img_list = [Image.open(os.path.join(data_repo,set,img)) for img in img_list]
    return img_list

# (a) First convert an image to grey level and re-scale the intensity to [0,31]
def convert_grey(img):
    '''
    Convert an image to grey
    Parameters:
        1. img: original image
    Return:
        1. img_grey: grey image

    '''
    

    # TODO: Add your code here
    if isinstance(img,Image.Image):
        img = np.array(img)
    # First convert to array, so that it is easy to operate down_sample() and rescale()
    
    # Check img_grey
    if len(img.shape) == 2:  
        return img
    
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_grey

def rescale(img_grey):
    '''
    Rescale the intensity to [0,31]
    Parameters:
        1. img_grey: grey image
    Return:
        1. scale_img_grey: scaled grey image

    '''
    # scale_img_grey = img_grey # Need to be changed
    # TODO: Add your code here
    scale_img_grey = np.floor((img_grey / 255) * 31).astype(np.uint8)
    return scale_img_grey


# (b) Convolve the images with a horizontal gradient filter ∇xI
def gradient_filter(img):
    '''
    This function is used to calculate horizontal gradient
    
    Parameters:
        1. img: img for calculating horizontal gradient 
    
    Return:
        1. img_dx: an array of horizontal gradient

    >>> img = np.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> gradient_filter(img)
    array([[1, 1],
           [1, 1],
           [1, 1]])
    '''
    
    img = img.astype(np.float32)

    img_dx = img[:, 1:] - img[:, :-1]
    return img_dx

def plot_Hz(img_dx,log = False):
    '''
    This function is used to plot the histogram of horizontal gradient
    '''
    # clear previous plot
    hz, bins_edge = np.histogram(img_dx, bins=list(range(-31, 31)))
    hz = hz/np.sum(hz)
    epsilon = 1e-5
    if log:
        plt.plot(bins_edge[:-1], np.log(hz+epsilon), color = color_blue,label="log Histogram")
    else:
        plt.plot(bins_edge[:-1], hz, color = color_blue,label="Histogram")
    return hz, bins_edge


def compute_mean_variance_kurtosis(img_dx):
    '''
    Compute the mean, variance, and kurtosis 

    Parameters:
        1. img_dx: an array of horizontal gradient (2D numpy array)

    Return:
        1. mean: mean of the horizontal gradient
        2. variance: variance of the horizontal gradient
        3. kurtosis: kurtosis of the horizontal gradient
    '''
    mean = np.mean(img_dx)
    
    variance = np.var(img_dx)
    
    kurtosis_ = scipy.stats.kurtosis(img_dx, fisher=False, bias=False)  # Use bias=False for unbiased estimate
    
    return mean, variance, kurtosis_


def GGD(x, sigma, gammar):
    ''' 
    pdf of GGD
    Parameters:
        1. x: input
        2. sigma: σ
        3. gammar: γ
    Note: The notation of x,σ,γ is the same as the document
    Return:
        1. y: pdf of GGD

    '''
    # TODO: Add your code here
    # NOTE according to the standard equation 
    sigma = max(sigma, 1e-5)  # Ensure sigma is not zero
    gammar = max(gammar, 1e-5)  # Ensure gammar is not zero
    
    coeff = gammar / (2 * sigma * scipy.special.gamma(1 / gammar))
    exp_part = np.exp(-(np.abs(x) / sigma) ** gammar)
    y = coeff * exp_part
    return y


def fit_GGD(hz, bins_edge):
    '''
    Fit the histogram to a Generalized Gaussian Distribution (GGD), and report the fittest sigma and gamma
    Parameters:
        1. hz: histogram of the horizontal gradient
        2. bins_edge: bins_edge of the histogram
    Return:
        None
    '''
    # fit the histogram to a generalized gaussian distribution

    datax = bins_edge[:-1]
    datay = hz
    
    init_params = [5,2]
    
    def GGD_fit_func(x,sigma,gammar):
        return GGD(x,sigma,gammar)
    
    popt, pcov = curve_fit(GGD_fit_func,datax,datay,p0 = init_params)
    
    sigma, gammar = popt
    
    print(f"After Fitted: Sigma: {sigma} ; Gammar: {gammar}")
    fitted_curve = GGD(datax, sigma, gammar)
    plt.plot(datax,fitted_curve,color = color_green,label = 'Fitted GGD')


    return sigma, gammar


def plot_Gaussian(mean,variance):
    ''' 
    Plot the Gaussian distribution using the mean and the variance
    Parameters:
        1. mean: mean of the horizontal gradient
        2. variance: variance of the horizontal gradient
    Return:
        None

    '''
    x = np.linspace(-31,31,500)

    # y = np.zeros(x.shape) # Need to be changed

    # TODO: Add your code here

    # y: value of pdf of Gassian distribution corresponding to x
    sigma = np.sqrt(variance)
    y = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(- (x - mean)**2 / (2 * sigma**2))

    plt.plot(x, y,color = color_green, label="Gaussian")
    return 

def plot_log_Gaussian(mean,variance):
    ''' 
    Plot the Gaussian distribution using the mean and the variance
    Parameters:
        1. mean: mean of the horizontal gradient
        2. variance: variance of the horizontal gradient
    Return:
        None

    '''
    x = np.linspace(-31,31,500)

    # y = np.zeros(x.shape) # Need to be changed

    # TODO: Add your code here

    # y: value of pdf of Gassian distribution corresponding to x
    sigma = np.sqrt(variance)
    epsilon = 1e-5
    y = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(- (x - mean)**2 / (2 * sigma**2))

    plt.plot(x, np.log(y+epsilon),color = color_green, label="log Gaussian")
    return 

def downsample(image):
    ''' 
    Downsample our images
    Parameters:
        1. image: original image
    Return:
        1. processed_image: downsampled image
    '''
    # processed_image = image # Need to be changed
    # TODO: Add your code here

    h, w = image.shape[:2]
    processed_h, processed_w = h//2, w//2
    processed_image = np.zeros((processed_h,processed_w),dtype = np.uint8)
    for i in range(processed_h):
        for j in range(processed_w):
            block = image[i*2:i*2+2 , j*2 :j*2+2]
            processed_image[i,j] = np.mean(block)
            
    
    # Averaging every 4 pixels

    return processed_image


def main():
    '''
    This is the main function
    '''
    # read img to img list
    # Notice: img_list is a list of image
    img_list = [read_img_list(set) for set in set_repo]
    # set_repo refers to the three sets we'll handle
    for idx1,set in enumerate(set_repo):
        img_dx_list = []
        img_dx_2_list = []
        img_dx_4_list = []
        for idx2,img in enumerate(img_list[idx1]):
            if set == 'setC':
                img_grey = convert_grey(img)
                img_2_grey = downsample(img_grey)
                img_4_grey = downsample(img_2_grey)
                # NOTE setC is random image, so it is not necessary for the rescaling operation
            else:
                img_grey = convert_grey(img)
                img_2_grey = downsample(img_grey)
                img_4_grey = downsample(img_2_grey)

                img_grey = rescale(img_grey)
                img_2_grey = rescale(img_2_grey)
                img_4_grey = rescale(img_4_grey)

            img_dx_list.append(gradient_filter(img_grey).flatten())
            img_dx_2_list.append(gradient_filter(img_2_grey).flatten())
            img_dx_4_list.append(gradient_filter(img_4_grey).flatten())
        img_dx = np.concatenate(img_dx_list)
        img_dx_2 = np.concatenate(img_dx_2_list)
        img_dx_4 = np.concatenate(img_dx_4_list)


        # plot histogram and log histogram
        print('--'*20)

        plt.clf()
        hz, bins_edge = plot_Hz(img_dx)
        # compute mean, variance and kurtosis
        mean, variance, kurtosis = compute_mean_variance_kurtosis(img_dx)
        print(f"set: {set}")
        print(f"mean: {mean}, variance: {variance}, kurtosis: {kurtosis}")

        # fit the histogram to a generalized gaussian distribution
        fit_GGD(hz, bins_edge)

        # plot the Gaussian distribution using the mean and the variance
        plot_Gaussian(mean,variance)

        plt.savefig(f"./pro1_result/histogram/{set}.png")

        # plot log histogram

        plt.clf()
        plot_Hz(img_dx,log=True)
        # save the histograms
        plt.savefig(f"./pro1_result/log_histogram/{set}.png")

        # plot the downsampled images histogram
        plt.clf()
        plot_Hz(img_dx)
        plt.savefig(f"./pro1_result/downsampled_histogram/original_{set}.png")

        plt.clf()
        plot_Hz(img_dx_2)
        plt.savefig(f"./pro1_result/downsampled_histogram/2_{set}.png")

        plt.clf()
        plot_Hz(img_dx_4)
        plt.savefig(f"./pro1_result/downsampled_histogram/4_{set}.png")


def main_report():
    """
    For report, I mainly change the plot order of the figures, and change related name methods.
    I also add more results, e.g. down-sampling 3 times,
    So I have opened a new main function
    """
    
    # read img to img list
    # Notice: img_list is a list of image
    img_list = [read_img_list(set) for set in set_repo]
    # set_repo refers to the three sets we'll handle
    for idx1,set in enumerate(set_repo):
        img_dx_list = []
        img_dx_2_list = []
        img_dx_4_list = []
        img_dx_8_list = []
        for idx2,img in enumerate(img_list[idx1]):
            if set == 'setC':
                img_grey = convert_grey(img)
                img_2_grey = downsample(img_grey)
                img_4_grey = downsample(img_2_grey)
                img_8_grey = downsample(img_4_grey)
                # NOTE setC is random image, so it is not necessary for the rescaling operation
            else:
                img_grey = convert_grey(img)
                img_2_grey = downsample(img_grey)
                img_4_grey = downsample(img_2_grey)
                img_8_grey = downsample(img_4_grey)
        
                

                img_grey = rescale(img_grey)
                img_2_grey = rescale(img_2_grey)
                img_4_grey = rescale(img_4_grey)
                img_8_grey = rescale(img_8_grey)

            img_dx_list.append(gradient_filter(img_grey).flatten())
            img_dx_2_list.append(gradient_filter(img_2_grey).flatten())
            img_dx_4_list.append(gradient_filter(img_4_grey).flatten())
            img_dx_8_list.append(gradient_filter(img_8_grey).flatten())
                     
        img_dx = np.concatenate(img_dx_list)
        img_dx_2 = np.concatenate(img_dx_2_list)
        img_dx_4 = np.concatenate(img_dx_4_list)
        img_dx_8 = np.concatenate(img_dx_8_list)


        # plot histogram and log histogram
        print('--'*20)
        
        # NOTE Problem 1 

        plt.clf()
        hz, bins_edge = plot_Hz(img_dx)
        os.makedirs('./pro1_result/new_report/vanilla_histogram',exist_ok=True)
        plt.savefig(f"./pro1_result/new_report/vanilla_histogram/{set}.pdf")
        
        
        plt.clf()
        plot_Hz(img_dx,log=True)
        # save the histograms
        os.makedirs('./pro1_result/new_report/vanilla_log_histogram',exist_ok=True)
        plt.savefig(f"./pro1_result/new_report/vanilla_log_histogram/{set}.pdf")
        
        # NOTE Problem 2
        
        # compute mean, variance and kurtosis
        mean, variance, kurtosis = compute_mean_variance_kurtosis(img_dx)
        print(f"set: {set}")
        print(f"mean: {mean}, variance: {variance}, kurtosis: {kurtosis}")
        
        # NOTE Problem 3
        plt.clf()
        # fit the histogram to a generalized gaussian distribution
        fit_GGD(hz, bins_edge)
        os.makedirs('./pro1_result/new_report/fitGGD_histogram',exist_ok=True)
        plt.savefig(f"./pro1_result/new_report/fitGGD_histogram/{set}.pdf")
        
        
        # NOTE problem 4 
        plt.clf()

        # plot the Gaussian distribution using the mean and the variance
        plot_Gaussian(mean,variance)
        plot_Hz(img_dx)
        os.makedirs('./pro1_result/new_report/gaussian_histogram/',exist_ok=True)
        plt.savefig(f"./pro1_result/new_report/gaussian_histogram/{set}.pdf")

        plot_log_Gaussian(mean,variance)
        plot_Hz(img_dx,log=True)
        os.makedirs('./pro1_result/new_report/gaussian_log_histogram/',exist_ok=True)
        plt.savefig(f"./pro1_result/new_report/gaussian_log_histogram/{set}.pdf")

        # plot log histogram
        
        # NOTE Problem 5

        # plot the downsampled images histogram
        plt.clf()
        plot_Hz(img_dx)
        plot_Hz(img_dx,log=True)
        os.makedirs('./pro1_result/new_report/downsampled_histogram/',exist_ok=True)
        plt.savefig(f"./pro1_result/new_report/downsampled_histogram/original_{set}.pdf")

        plt.clf()
        plot_Hz(img_dx_2)
        plot_Hz(img_dx_2,log=True)
        plt.savefig(f"./pro1_result/new_report/downsampled_histogram/2_{set}.pdf")

        plt.clf()
        plot_Hz(img_dx_4)
        plot_Hz(img_dx_4,log=True)
        plt.savefig(f"./pro1_result/new_report/downsampled_histogram/4_{set}.pdf")

        plt.clf()
        plot_Hz(img_dx_8)
        plot_Hz(img_dx_8,log=True)
        plt.savefig(f"./pro1_result/new_report/downsampled_histogram/8_{set}.pdf")

if __name__ == '__main__':
    main()
    # main_report()

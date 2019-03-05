import scipy
from scipy import signal
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import matplotlib.colors as colors


def read_img(path):
    return cv2.imread(path, 0)


def save_img(img, path):
    cv2.imwrite(path,img)
    print(path, "is saved!")


def display_img(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Input- image: H x W
    #        kernel: h x w
    # Output- convolve: H x W
    output = signal.convolve2d(image, kernel, boundary='symm', mode = 'same')
    # TODO: You can use the function scipy.signal.convolve2d().
    return output


def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    # TODO: Fix kx, ky
#    kx = 0.5*np.array([[0,0,0],[1,0,-1],[0,0,0]])  # 1 x 3
#    ky = 0.5*np.array([[0,1,0],[0,0,0],[0,-1,0]])  # 3 x 1
    kx = 0.5*np.array([[1,0,-1]])
    ky = 0.5*np.array([[1],[0],[-1]])

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix**2 + Iy**2)

    return grad_magnitude, Ix, Iy


def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    # TODO: Use convolve() to complete the function
    Gx, Gy, grad_magnitude = None, None, None
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])  # 1 x 3
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])  # 3 x 1

    Gx = convolve(image, kx)
    Gy = convolve(image, ky)
    
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)
    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=[0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W
    # You are encouraged not to use sobel_operator() in this function.

    # TODO: Use convolve() to complete the function
    output = []
    for i in range(len(angles)):
        kernal = kernal_steerable_filter(angles[i])
        output.append(convolve(image, kernal))        
    return output


def kernal_steerable_filter(alpha):
    return np.array([[np.cos(alpha)+np.sin(alpha), 2*np.sin(alpha), -np.cos(alpha)+np.sin(alpha)],
                      [2*np.cos(alpha), 0 , -2*np.cos(alpha)],
                      [np.cos(alpha)-np.sin(alpha), -2*np.sin(alpha), -np.cos(alpha)-np.sin(alpha)]])

    
def image_patches(image, patch_size=(16,16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N 
    # Output- results: a list of images of size M x N

    # TODO: Use slicing to complete the function
    if (len(image.shape)==2):
        m,n = image.shape
    else:
        m, n, _ = image.shape
    h_num = m//patch_size[0]
    w_num = n//patch_size[1]
    output = []
    for i in range(h_num):
        for j in range(w_num):
            h_start, h_end = i*patch_size[0], (i+1)*patch_size[0]
            w_start, w_end = j*patch_size[1], (j+1)*patch_size[1]
            if (len(image.shape)==2): 
                output.append(image[h_start:h_end, w_start:w_end])
            else:
                output.append(image[h_start:h_end, w_start:w_end, :])
    return output


def gaussian(x,y,sigma):
    return 1/(2*np.pi*sigma)*np.exp(-(x**2+y**2)/(2*sigma))
    

def gaussian_kernal(sigma, kernal_size=(3,3)):
    offset_h=kernal_size[0]//2
    offset_w=kernal_size[1]//2
    result = np.ones(kernal_size)
    for i in range(kernal_size[0]):
        for j in range(kernal_size[1]):
            result[i,j] = gaussian(i-offset_h, j-offset_w, sigma)
    return result
    

def gaussian1d(x, sigma):
    return 1/np.sqrt(2*np.pi*sigma) * np.exp(-x**2/(2*sigma))

def gaussian1d_prime(x, sigma):
    return -1*x/(sigma*np.sqrt(2*np.pi*sigma)) * np.exp(-x**2/(2*sigma))

def gaussian1d_prime_prime(x, sigma):
    return -1/(sigma*np.sqrt(2*np.pi*sigma)) * (np.exp(-x**2/(2*sigma))-x**2/sigma*np.exp(-x**2/(2*sigma)))


def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Image Patches #####
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # Q1
    patch_size = (16,16)
    patches = image_patches(img, patch_size)
    # TODO choose a few patches and save them
    chosen_patches = patches[200];
    save_img(chosen_patches, "./image_patches/q1_patch.png")
#    display_img(chosen_patches)

    # Q2: No code

    ##### Gaussian Filter #####
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # Q1: No code

    # Q2

    # TODO: Calculate the kernel described in the question.  There is tolerance for the kernel.
    sigma = 1/(2*np.log(2))
    kernel_gaussian = gaussian_kernal(sigma,(3,3))

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # Q3
    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")
#    display_img(edge_detect)
    print("Gaussian Filter is done. ")
    
    
    
    
    ########################

    ##### Sobel Operator #####
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Q1: No code

    # Q2
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")
    
    
    # Q3
    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    ########################





    #####LoG Filter#####
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # Q1
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = 0.04*np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    print(img.shape)
    # Q2: No code
#    sigma = 1/(2*np.log(2))
    sigma = 2
    sigma1 = sigma/2.5
    sigma2 = sigma*2.5
    x = np.linspace(-10,10,200)
    y = gaussian1d_prime_prime(x, sigma)
    y1 = gaussian1d(x, sigma1)
    y2 = gaussian1d(x, sigma2)
#    plt.figure()
#    plt.plot(x,sigma*y,'b',label = 'LOG')
#    plt.plot(x, (y2-y1),'r', label = 'DoG')
#    plt.legend()
    
    print("LoG Filter is done. ")
    ########################


if __name__ == "__main__":
    main()

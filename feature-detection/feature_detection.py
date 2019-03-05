import os
import cv2
import numpy as np
import filters
from matplotlib import pyplot as plt
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
    

def harris_corner_detector(image, x_offset=2, y_offset=2, window_size=(5,5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return an heatmap image where every pixel is the harris
    # corner detector score for that pixel.
    # OR, do this with gradients (think Sobel operator) and
    # the structure tensor. 
    # Input- image: H x W
    #        x_offset: a scalar
    #        y_offset: a scalar
    #        window_size: a scalar tuple M, N 
    # Output- results: a image of size H x W

    # TODO: Implement and run a harris corner detector on the image
    output = []
    alpha = 0.04
    H, W = image.shape
    output = np.zeros((H-window_size[0]+1, W-window_size[1]+1))
    Ix, Iy, _ = filters.sobel_operator(image)
    sigma = 1.5/(np.log(2))
    weight = filters.gaussian_kernal(sigma,window_size)
    for i in range(H-window_size[0]+1):
        for j in range(W-window_size[1]+1):
            h_start, h_end = i, i+window_size[0]
            w_start, w_end = j, j+window_size[1]
            temp11 = np.sum(weight*Ix[h_start:h_end, w_start:w_end]**2)
            temp12 = np.sum(weight*Ix[h_start:h_end, w_start:w_end]*Iy[h_start:h_end, w_start:w_end])
            temp22 = np.sum(weight*Iy[h_start:h_end, w_start:w_end]**2)
            M = np.array([[temp11, temp12],[temp12, temp22]])
            output[i,j] = np.linalg.det(M) - alpha*np.trace(M)**2
    plt.figure()
    plt.imshow(output, cmap = plt.cm.hot)
    plt.colorbar()
    plt.show()
    return output


def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

#    display_img(img)
    
    ##### Feature Detection #####  
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

#    filters.display_img(img)    
    
    
    harris_corner_image = harris_corner_detector(img)
    save_img(harris_corner_image, "./feature_detection/q1.png")
    
    
    
#    filters.display_img(harris_corner_image)

if __name__ == "__main__":
    main()

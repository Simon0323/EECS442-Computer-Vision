from skimage import io 
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

def task_1(img,filename, enable_save):
    #img = cv2.imread('prokudin-gorskii/01112v.jpg', cv2.IMREAD_GRAYSCALE)
    m, n = img.shape
    m -= m % 3
    m_new = int(m/3)
    img_b = img[0:m_new,:]
    img_g = img[m_new:2*m_new,:]
    img_r = img[2*m_new:m,:]
    img_color = np.dstack([img_b, img_g, img_r])  # using opencv save image also need to use BGR
    #save the image
    if (enable_save):
        saved_filename = filename + "_combined.jpg"
        cv2.imwrite(saved_filename, img_color)
    # plot image 
    #img_color_plot = np.dstack([img_r, img_g, img_b])  
    #plt.imshow(img_color_plot)
    return img_color
  
    
def task_2(img, maxstep, filename, enable_save):
    offset_x_b, offset_y_b, offset_x_g, offset_y_g = 0, 0, 0, 0
    max_prod1, max_prod2 = 0, 0
    m,n = img.shape[0], img.shape[1] 
    x_start, x_end, y_start, y_end = 8+maxstep, m-maxstep-10, 8+maxstep, n-maxstep-10
    img_b, img_g, img_r = img[:,:,0], img[:,:,1], img[:,:,2] # use red as stationary
    v_r = np.reshape(img_r[x_start:x_end,y_start:y_end],-1)
    v_b = np.reshape(img_b[x_start:x_end,y_start:y_end], -1)
    v_g = np.reshape(img_g[x_start:x_end,y_start:y_end], -1)
    mean_r = np.mean(v_r)
    mean_b = np.mean(v_b)
    mean_g = np.mean(v_g)
    for i in range(-maxstep,1+maxstep):
        for j in range(-maxstep,1+maxstep):
            v_b = np.reshape(img_b[x_start+i:x_end+i,y_start+j:y_end+j], -1)
            v_g = np.reshape(img_g[x_start+i:x_end+i,y_start+j:y_end+j], -1)
            #temp1 = np.linalg.norm(v_b-v_r)
            #temp2 = np.linalg.norm(v_g-v_r)
            temp1 = np.dot((v_b-mean_b),(v_r-mean_r))
            temp2 = np.dot((v_g-mean_g),(v_r-mean_r))
            if max_prod1 == 0 or temp1 > max_prod1:
                offset_x_b, offset_y_b = i, j
                max_prod1 = copy.deepcopy(temp1)
            if max_prod2 == 0 or temp2 > max_prod2:
                offset_x_g, offset_y_g = i, j
                max_prod2 = copy.deepcopy(temp2)

    print(offset_x_b, offset_y_b, offset_x_g, offset_y_g)
    img_final_b = img_b[x_start+offset_x_b:x_end+offset_x_b,y_start+offset_y_b:y_end+offset_y_b]
    img_final_g = img_g[x_start+offset_x_g:x_end+offset_x_g,y_start+offset_y_g:y_end+offset_y_g]
    img_final_r = img_r[x_start:x_end, y_start:y_end]
    img_final = np.dstack([img_final_b, img_final_g, img_final_r])
    # use to save the image 
    if (enable_save):
        saved_filename = filename + "_aglined.jpg"
        cv2.imwrite(saved_filename, img_final)
    # plot
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # the original graph
    img_final_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    plt.imshow(img_final_rgb)
    return img_final, offset_x_b, offset_y_b, offset_x_g, offset_y_g
    

def task_3(img, file_name, enable_save):
    m, n = img.shape[0], img.shape[1]
    print('m = ', m, 'n = ', n)
    if m%2!=0: m-=1
    if n%2!=0: n-=1
    indexr = range(0,m,2)
    indexc = range(0,n,2)
    img_half = copy.deepcopy(img)
    img_half = np.delete(img_half,indexr, 0)
    img_half = np.delete(img_half,indexc, 1)
    
    img_temp, off_x_b, off_y_b, off_x_g, off_y_g = task_2(img_half, 15, "Hello", False)
    img_show = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
    plt.imshow(img_show)
    
    x_l, y_l = max([0,-off_x_b*2, -off_x_g*2]), max([0, -off_y_b*2,-off_y_g*2])
    x_h, y_h = min([m, m-off_x_b*2, m-off_x_g*2]), min([n, n-off_y_b*2, n-off_y_g*2])
    
    img_b, img_g, img_r = img[:,:,0], img[:,:,1], img[:,:,2]
    img1 = np.dstack((img_b[x_l + 2*off_x_b : x_h + 2*off_x_b, y_l + 2*off_y_b : y_h + 2*off_y_b], 
                      img_g[x_l + 2*off_x_g : x_h + 2*off_x_g, y_l + 2*off_y_g : y_h + 2*off_y_g], 
                      img_r[x_l : x_h, y_l : y_h]))
    
    img_final, off_x_b1, off_y_b1, off_x_g1, off_y_g1 = task_2(img1, 5, "Hello", False)
    img_final_show = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    final_x_b, final_y_b = off_x_b*2 + off_x_b1, off_y_b*2 + off_y_b1
    final_x_g, final_y_g = off_x_g*2 + off_x_g1, off_y_g*2 + off_y_g1 
    print("Final Offset = ", final_x_b, final_y_b, final_x_g, final_y_g)
    if (enable_save):
        saved_filename = filename + "_aglinedByTask3.jpg"
        cv2.imwrite(saved_filename, img_final)
    plt.imshow(img_final_show)



if __name__ == "__main__":
    # read the original picture and split it into a colored picture
    enable_save = False
    filename = "vancouver_tableau"
    print (filename)
    #directory = "prokudin-gorskii/"
    directory = ""
    img0 = cv2.imread(directory+filename+".jpg", cv2.IMREAD_GRAYSCALE)
    
    img = task_1(img0, filename, enable_save)
    
    #img2 = task_2(img, 15, filename, enable_save)
        
    task_3(img, filename, True)




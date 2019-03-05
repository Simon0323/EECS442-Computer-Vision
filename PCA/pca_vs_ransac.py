#!/usr/bin/env python
import utils
import numpy
import time
import random
import matplotlib
###YOUR IMPORTS HERE###
import pca_ransac_fun
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')
    pca_outlier=[]
    ransac_outlier=[]
    pca_error = []
    ransca_error = []
    num_tests = 10
    fig = None
    for i in range(0, num_tests):
        pc = add_some_outliers(pc, 10) #adding 10 new outliers for each test
        fig = utils.view_pc([pc])
        ###YOUR CODE HERE###
        print i
        theta = 0.08

        t1 = time.clock()
        error_pca, num_pca = pca_ransac_fun.pca(pc, theta, i, num_tests)
        pca_error.append(error_pca)
        pca_outlier.append(num_pca)
        t2 = time.clock()
        print 'Time_PCA ', t2-t1, '\n'

        t1 = time.clock()
        error_ransac, num_ransac = pca_ransac_fun.ransac(pc, theta, i, num_tests)
        ransca_error.append(error_ransac)
        ransac_outlier.append(num_ransac)
        t2 = time.clock()
        print 'Time_Ransac ', t2-t1, '\n'

        #this code is just for viewing, you can remove or change it
        #raw_input("Press enter for next test:")
        matplotlib.pyplot.close(fig)
    plt.subplot(2,1,1)
    plt.plot(pca_outlier, pca_error, 'o-')
    plt.title('PCA')
    plt.title('PCA')
    plt.xlabel('Number of outlier')
    plt.ylabel('Error')

    plt.subplot(2,1,2)
    plt.plot(ransac_outlier, ransca_error, 'o-')
    plt.title('Ransac')
    plt.xlabel('Number of outlier')
    plt.ylabel('Error')
    plt.show()
    ###YOUR CODE HERE###

    raw_input("Press enter to end:")


if __name__ == '__main__':
    main()

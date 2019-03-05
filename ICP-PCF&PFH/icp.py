#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import icp_fun
import copy
import random
import time
###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target
    shape_p = numpy.shape(pc_source)
    shape_q = numpy.shape(pc_target)
    pc_p = numpy.reshape(copy.deepcopy(pc_source), (shape_p[0], shape_p[1])).T
    pc_q = numpy.reshape(copy.deepcopy(pc_target), (shape_q[0], shape_q[1])).T
    pc_p_origin = copy.deepcopy(pc_p)
    thre = 100
    ite_num = 0
    error = 2
    error_all = []
    #while ite_num < thre:
    t1 = time.clock()
    while error > 0.001:
        pc_q = icp_fun.correspondence(pc_p, pc_q)
        pc_p, error, temp1, temp2 = icp_fun.icp_cal(pc_p, pc_q, ite_num)
        ite_num += 1
        if ite_num % 25 == 0:
            print 'Iteration number: \n', ite_num
        error_all.append(error)
    t2 = time.clock()
    print 'Time cost: ', t2-t1
    #the final R and T
    temp1, temp2, R, t = icp_fun.icp_cal(pc_p_origin, pc_q, 0)
    print R, '\n', t
    #plot for error and iteration
    ite_all = numpy.reshape(range(ite_num), (ite_num, ))
    plt.plot(ite_all, error_all, 'o-')
    plt.title('Error vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()
    #plot the final point cloud
    pc_result = numpy.reshape(pc_p.T, shape_p)
    result = []
    for i in range(len(pc_source)):
        result.append(pc_result[i, :].T)
    utils.view_pc([result, pc_target], None, ['b', 'r'], ['o', '^']) #after icp
    plt.axis([-0.15, 0.15, -0.15, 0.15])
    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^']) #before icp
    plt.axis([-0.15, 0.15, -0.15, 0.15])
    ###YOUR CODE HERE###

    raw_input("Press enter to end:")


if __name__ == '__main__':
    main()

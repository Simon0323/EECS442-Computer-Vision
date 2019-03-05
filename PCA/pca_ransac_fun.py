#!/usr/bin/env python
import utils
import numpy
###YOUR IMPORTS HERE###
import random
from operator import itemgetter
import copy
import matplotlib
###YOUR IMPORTS HERE###


def ransac(pc, theta, i, limit):
    num_thred = 140
    shape_pc = numpy.shape(pc)
    num_point = shape_pc[0]
    data = numpy.matrix(numpy.reshape(pc,(shape_pc[0], shape_pc[1])))
    #Fit a plane to the data using ransac
    num_iteration = 500
    min_err = 100
    flag = 0
    for i in range(num_iteration):
        index1, index2, index3 = random.sample(range(num_point), 3)
        reference_point = data[index1][:]
        v1 = data[index2][:]-data[index1][:]
        v2 = data[index3][:]-data[index1][:]
        dir_plane = numpy.cross(v1, v2)
        if numpy.linalg.norm(dir_plane) == 0:
            continue
        else:
            dir_plane = dir_plane / numpy.linalg.norm(dir_plane)
        temp = numpy.absolute(dir_plane*(data - reference_point).T)
        index_inlier = numpy.where(temp < theta)[1]
        num_inlier = len(index_inlier)
        if num_inlier > num_thred:
            data_inlier = data[index_inlier, :]
            mean_data_in = numpy.mean(data_inlier, 0)
            data_in_zero = data_inlier - mean_data_in
            u, s, vh = numpy.linalg.svd(numpy.matrix(data_in_zero).T*numpy.matrix(data_in_zero)/(num_inlier-1))
            if s[2]/num_inlier < min_err:
                min_err = s[2]/num_inlier
                plane_dir = vh[:][2]
                plane_point = mean_data_in
                flag = 1
    #Show the resulting point cloud
    temp = numpy.absolute(plane_dir*(data - plane_point).T)
    index_inlier = numpy.where(temp < theta)[1]
    index_outlier = range(num_point)
    index_outlier = numpy.delete(index_outlier, index_inlier)
    data_inlier_dis = copy.deepcopy(itemgetter(*index_inlier)(pc))
    data_outlier_dis = copy.deepcopy(itemgetter(*index_outlier)(pc))
    if i == limit-1:
        fig = utils.view_pc([data_outlier_dis])
        fig = utils.view_pc([data_inlier_dis], fig, ['r'])
        #Draw the fitted plane
        if flag == 1:
            fig = utils.draw_plane(fig, plane_dir.T, plane_point.T, (0.1, 1, 0.1, 0.5), [-0.5, 1], [-0.5,1.3])
        raw_input("Press enter to end:")
        matplotlib.pyplot.close(fig)
    outlier_num = shape_pc[0]-len(index_inlier)
    error = (temp[:, index_inlier] * temp[:, index_inlier].T / len(index_inlier))[0, 0]
    print 'error=', error
    print 'num_outlier', outlier_num
    return error, outlier_num


def pca(pc, theta, i, limit):
    #Rotate the points to align with the XY plane
    shape_pc = numpy.shape(pc)
    data = numpy.reshape(copy.deepcopy(pc), (shape_pc[0], shape_pc[1]))
    pc_mean = numpy.mean(data, 0).T
    pc_present = data - pc_mean
    q = numpy.matrix(pc_present.T) * numpy.matrix(pc_present)/199
    u, s, vh = numpy.linalg.svd(q)
    dir_plan = numpy.matrix(vh)[:][2]

    temp = numpy.absolute(dir_plan*(data - pc_mean).T)
    index_inlier = numpy.where(temp < theta)[1]
    index_outlier = range(shape_pc[0])
    index_outlier = numpy.delete(index_outlier, index_inlier)
    data_inlier_dis = copy.deepcopy(itemgetter(*index_inlier)(pc))
    data_outlier_dis = copy.deepcopy(itemgetter(*index_outlier)(pc))
    if i == limit-1:
        fig = utils.view_pc([data_outlier_dis])
        fig = utils.view_pc([data_inlier_dis], fig, ['r'])
        fig = utils.draw_plane(fig, dir_plan.T, numpy.matrix(pc_mean).T, (0.1, 1, 0.1, 0.5), [-0.5, 1], [-0.5,1.3])
        raw_input("Press enter to end:")
        matplotlib.pyplot.close(fig)
    outlier_num = shape_pc[0]-len(index_inlier)
    error = (temp[:, index_inlier] * temp[:, index_inlier].T / len(index_inlier))[0, 0]
    print 'error=', error
    print 'num_outlier', outlier_num
    return error, outlier_num

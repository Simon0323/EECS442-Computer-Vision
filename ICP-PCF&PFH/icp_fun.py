#!/usr/bin/env python
import numpy
import copy
import random
import time

#closest point use loop
def closepoint(pc, index, targetconfig):  #pc is 3*m matrix
    err = pc - targetconfig
    index_min = 0
    err_min = numpy.linalg.norm(err[:, index[index_min]])
    for i in range(len(index)):
        temp = numpy.linalg.norm(err[:, index[i]])
        if temp < err_min:
            index_min = i
            err_min = temp
    index_close = index[index_min]
    del index[index_min]
    return index_close
#closest point use matrix
def closedis_m(pc, index, targetconfig):
    pc_present = copy.deepcopy(pc[:, index])
    err = numpy.linalg.norm(pc_present - targetconfig, axis=0)
    index_temp = numpy.argmin(err)
    index_close = index[index_temp]
    del index[index_temp]
    return index_close

#calculate feature for the two point cloud
def feature_calculation(pc, thred_neighbor):  #PC is a row matrix
    len_data = numpy.shape(pc)[1]
    for i in range(len_data):
        neighbor = find_neighbor(pc, i, thred_neighbor)
        x = neighbor*neighbor.T
        u, s, vh = numpy.linalg.svd(x)
        temp = numpy.ones((4, 1))
        temp[0:3, :] = numpy.matrix(copy.deepcopy(vh)).T[2][:].T
        temp[3][0] = s[2]/(s[0]+s[1]+s[2])
        if i == 0:
            feature = copy.deepcopy(temp)
        else:
            feature = numpy.hstack((feature, copy.deepcopy(temp)))
    return feature

#finde the neighbor in the distance less than thred
def find_neighbor(pc, index, thred):
    target = numpy.reshape(copy.deepcopy(pc[:, index]), (3, 1))
    error = numpy.linalg.norm(pc - target, axis=0)
    index = numpy.where(error < thred)
    neighbor = numpy.matrix(copy.deepcopy(pc[:, index[0]]))
    return neighbor


#just use distance of the points
def correspondence(pc_p, pc_q):
    index_aviliable = range(numpy.shape(pc_q)[1])
    index_all = range(numpy.shape(pc_p)[1])
    random.shuffle(index_all)
    pc_new = numpy.matrix(copy.deepcopy(pc_q))
    for i in index_all:
        index_temp = closedis_m(pc_q, index_aviliable, numpy.reshape(pc_p[:, i], (3, 1)))
        pc_new[:, i] = numpy.reshape(copy.deepcopy(pc_q[:, index_temp]), (3, 1))
    return pc_new

def icp_cal(pc_p, pc_q, ite_num):
    p_mean = numpy.reshape(numpy.mean(pc_p, 1), (3, 1))
    q_mean = numpy.reshape(numpy.mean(pc_q, 1), (3, 1))
    x = pc_p - p_mean
    y = pc_q - q_mean
    data = numpy.matrix(x)*numpy.matrix(y).T
    u, s, vh = numpy.linalg.svd(data)
    flip = numpy.linalg.det(vh.T*u.T)
    R = vh.T*numpy.matrix([[1, 0, 0], [0, 1, 0], [0, 0, flip]])*u.T
    t = q_mean - R*p_mean
    temp = numpy.matrix(R*pc_p + t - pc_q)
    error_matrix = numpy.reshape(numpy.linalg.norm(temp, axis=0), (numpy.shape(pc_p)[1], 1))
    error = numpy.matrix(error_matrix).T*numpy.matrix(error_matrix)
    #print error
    if ite_num > 40 and error > 0.1 and random.random() < 0.08:
        R = randomrotation()
    if error < 5:
        pc_p = numpy.matrix(R*pc_p + t)
    return pc_p, error[0, 0], R, t


#use PCF point cloud feature to calculate correspondence
def correspondence_PCF(pc_p, pc_q, area):
    feature_p = feature_calculation(pc_p, area)
    feature_q = feature_calculation(pc_q, area)
    index_aviliable = range(numpy.shape(pc_q)[1])
    random.shuffle(index_aviliable)
    for i in range(numpy.shape(pc_p)[1]):
        #index_temp = closepoint(feature_q, index_aviliable, numpy.reshape(feature_p[:, i], (4, 1)))
        index_temp = closedis_m(feature_q, index_aviliable, numpy.reshape(feature_p[:, i], (4, 1)))
        if i == 0:
            pc_new = numpy.reshape(copy.deepcopy(pc_q[:, index_temp]), (3, 1))
        else:
            pc_new = numpy.hstack((pc_new, numpy.reshape(copy.deepcopy(pc_q[:, index_temp]), (3, 1))))
    return pc_new


def randomrotation():
    beta = 0.8
    theta1 = beta*(2*numpy.pi*random.random()-numpy.pi)
    theta2 = beta*(2*numpy.pi*random.random()-numpy.pi)
    theta3 = beta*(2*numpy.pi*random.random()-numpy.pi)
    rot1 = numpy.matrix([[1, 0, 0],
                         [0, numpy.cos(theta1), -numpy.sin(theta1)],
                         [0, numpy.sin(theta1), numpy.cos(theta1)]])
    rot2 = numpy.matrix([[numpy.cos(theta2), 0, numpy.sin(theta2)],
                         [0, 1, 0],
                         [-numpy.sin(theta2), 0, numpy.cos(theta2)]])
    rot3 = numpy.matrix([[numpy.cos(theta3), -numpy.sin(theta3), 0],
                         [numpy.sin(theta3), numpy.cos(theta3), 0],
                         [0, 0, 1]])
    rot = rot1*rot2*rot3
    return rot


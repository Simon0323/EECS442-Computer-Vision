#!/usr/bin/env python
import utils
import numpy
###YOUR IMPORTS HERE###
import copy
import numpy
###YOUR IMPORTS HERE###


def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    fig = utils.view_pc([pc])
    #Rotate the points to align with the XY plane
    thredhold = 0.01
    pc = numpy.reshape(pc,(200,3))
    pc_mean = numpy.mean(pc, 0).T
    pc_present = pc - pc_mean
    q = numpy.matrix(pc_present.T) * numpy.matrix(pc_present)/199
    u, s ,vh = numpy.linalg.svd(q)
    x_new = (vh.T * numpy.matrix(pc_present.T)).T
    print 'w=\n', vh
    #Show the resulting point cloud
    pc_present_display = numpy.reshape(copy.deepcopy(pc_present), (200,3,1))
    utils.view_pc([pc_present_display])
    x_new_dis = numpy.reshape(numpy.asarray(copy.deepcopy(x_new)), (200,3,1))
    utils.view_pc([x_new_dis])
    #Rotate the points to align with the XY plane AND eliminate the noise
    # Show the resulting point cloud
    for i in range(len(s)):
        if s[i] < thredhold:
            vh_cut = (numpy.matrix(copy.deepcopy(vh)).T[:][0:i]).T
            print vh_cut
            x_new_cut = (vh_cut.T * numpy.matrix(pc_present.T)).T
            extra_col = numpy.zeros((200, 1))
            x_new_cut_dis = numpy.reshape(numpy.array(numpy.hstack((x_new_cut, extra_col))), (200,3,1))
            utils.view_pc([x_new_cut_dis])
            break

    dir_plan = numpy.matrix(vh)[:][2]
    fig = utils.draw_plane(fig, dir_plan.T, numpy.matrix(pc_mean).T, (0.1, 1, 0.1, 0.5), [-0.5, 1], [-0.5,1.3])
    ###YOUR CODE HERE###


    raw_input("Press enter to end:")


if __name__ == '__main__':
    main()

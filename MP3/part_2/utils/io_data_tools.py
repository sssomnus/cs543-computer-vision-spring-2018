"""Input and output helpers to load and preprocess data."""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_plot_data(root, img_name):
    # Load image as Image object and convert to grayscale.
    I1 = Image.open(root+'data/'+img_name+'1.jpg').convert('L')
    I2 = Image.open(root+'data/'+img_name+'2.jpg').convert('L')

    # Load matches.
    # this is a N x 4 file where the first two numbers of each row
    # are coordinates of corners in the first image and the last two
    # are coordinates of corresponding corners in the second image: 
    # matches(i,1:2) is a point in the first image
    # matches(i,3:4) is a corresponding point in the second image
    matches = np.loadtxt(root+'data/'+img_name+'_matches.txt')

    # Display two images side-by-side with matches.
    I3 = np.zeros((I1.size[1],I1.size[0]*2) )
    I3[:,:I1.size[0]] = I1;
    I3[:,I1.size[0]:] = I2;
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(float), cmap='gray')
    ax.plot(matches[:,0],matches[:,1],  '+r')
    ax.plot(matches[:,2]+I1.size[0],matches[:,3], '+r')
    ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 
            'r', linewidth=0.3)
    plt.axis('off')
    plt.savefig(root+'outputs/'+img_name+'_matches.svg', format='svg')
    plt.show()

    # Return Image object as numpy array.
    I1_arr = np.array(I1)
    I2_arr = np.array(I2)

    return I1_arr, I2_arr, matches


def plot_epipolar(matches, F, I2_arr, root, name):
    """Display second image with epipolar lines reprojected 
    from the first image."""

    # first, fit fundamental matrix to the matches
    # F = fit_fundamental(matches); # this is a function that you should write
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
    closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2_arr).astype(float), cmap='gray')
    ax.plot(matches[:,2],matches[:,3], 'or', markersize=2)
    ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
    ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g', linewidth=1)
    plt.axis('off')
    plt.savefig(root+'outputs/'+name+'.svg', format='svg')
    plt.show()


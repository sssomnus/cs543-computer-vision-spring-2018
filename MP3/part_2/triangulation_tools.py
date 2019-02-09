import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


def triangulate(P1, P2, matches):
    # Don't know why needs to transpose V, but it just works..
    U, s, V = np.linalg.svd(P1)
    center1 = V.T[:, -1]
    center1 = center1/center1[-1]
    
    U, s, V = np.linalg.svd(P2)
    center2 = V.T[:, -1]
    center2 = center2/center2[-1]
    
    # Convert on homogeneous.
    ones = np.ones((matches.shape[0], 1))
    points1 = np.concatenate((matches[:, 0:2], ones), axis=1)
    points2 = np.concatenate((matches[:, 2:4], ones), axis=1) 

    # Reconstruct 3D points.
    X_3d = np.zeros((matches.shape[0], 4))
    for i in range(matches.shape[0]):
        x1_cross_P1 = np.array([[0, -points1[i,2], points1[i,1]], 
                          [points1[i,2], 0, -points1[i,0]], 
                          [-points1[i,1], points1[i,0], 0]])
        x2_cross_P2 = np.array([[0, -points2[i,2], points2[i,1]], 
                          [points2[i,2], 0, -points2[i,0]], 
                          [-points2[i,1], points2[i,0], 0]])

        x_cross_P = np.concatenate((x1_cross_P1.dot(P1), x2_cross_P2.dot(P2)), 
                                   axis=0)
        
        # X_3d will become inf when I don't use the tmp var, I don't know why.
        U, s, V = np.linalg.svd(x_cross_P)
        temp = V.T[:, -1]
        temp = temp / temp[-1]
        X_3d[i] = temp

    return center1, center2, X_3d


def get_residual(P1, P2, X_3d, matches):
    # Project 3D points back to 2D and convert to homogeneous.
    projected1 = np.dot(P1, X_3d.T).T
    projected1 = projected1 / projected1[:, -1][:, np.newaxis]
    projected2 = np.dot(P2, X_3d.T).T
    projected2 = projected2 / projected2[:, -1][:, np.newaxis]
    # Compute residual.
    res1 = np.linalg.norm(projected1[:, 0:2]-matches[:, 0:2]) ** 2
    res2 = np.linalg.norm(projected2[:, 0:2]-matches[:, 2:4]) ** 2
    # avg_res = (res1 + res2) / 2 / matches.shape[0]

    avg_res1 = res1 / matches.shape[0]
    avg_res2 = res2 / matches.shape[0]
    
    return avg_res1, avg_res2


def plot_3d(center1, center2, X_3d):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], c='b', marker='o', alpha=0.6)
    ax.scatter(center1[0], center1[1], center1[2], c='r', marker='+')
    ax.scatter(center2[0], center2[1], center2[2], c='g', marker='+')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

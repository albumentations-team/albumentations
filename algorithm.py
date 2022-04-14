import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rotate_bounding_box(B, angle):

    h2 = B[2]/2
    w2 = B[3]/2

    X = []
    Y = []

    # generate coordinates for ellipse
    for i in range(360):
        X.append(w2 * math.sin(math.radians(i)) + B[0])
        Y.append(h2 * math.cos(math.radians(i)) + B[1])

    # combine X and Y coords
    points = list(zip(X,Y))

    # rotate points by the angle with respect to the origin
    points2 = rotate(points, origin=(B[0],B[1]), degrees=angle)

    X2 = [p[0] for p in points2]
    Y2 = [p[1] for p in points2]

    # # create bounding box for original ellipse
    # X_bounding_box = [max(X), max(X), min(X), min(X)]
    # Y_bounding_box = [max(Y), min(Y), min(Y), max(Y)]

    # # create bounding box for rotated ellipse
    # X2_bounding_box = [max(X2), max(X2), min(X2), min(X2)]
    # Y2_bounding_box = [max(Y2), min(Y2), min(Y2), max(Y2)]

    # draw original bounding box
    box1 = mpatches.Rectangle((min(X),min(Y)),max(X)-min(X),max(Y)-min(Y), 
                        fill = False,
                        color = "purple",
                        linewidth = 2)
    plt.gca().add_patch(box1)

    # draw rotated bounding box
    box2 = mpatches.Rectangle((min(X2),min(Y2)),max(X2)-min(X2),max(Y2)-min(Y2), 
                        fill = False,
                        color = "red",
                        linewidth = 2)
    plt.gca().add_patch(box2)
    
    # plot ellipse
    plt.scatter(X, Y)
    plt.scatter(X2, Y2)

    plt.show()


def main():
    rotate_bounding_box(B=[30,30,10,20],angle=30)

if __name__ == "__main__":
    main()
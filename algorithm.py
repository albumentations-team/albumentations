# WILL REMOVE THIS FILE ONCE THE PUSH REQUEST HAS BEEN APPROVED

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# def rotate(p, origin=(0, 0), degrees=0):
#     angle = np.deg2rad(degrees)
#     R = np.array([[np.cos(angle), -np.sin(angle)],
#                   [np.sin(angle),  np.cos(angle)]])
#     o = np.atleast_2d(origin)
#     p = np.atleast_2d(p)
#     return np.squeeze((R @ (p.T-o.T) + o.T).T)

def rotate_bounding_box(bbox, angle):

    x_min, y_min, x_max, y_max = bbox[:4]

    w2 = (x_max-x_min)/2
    h2 = (y_max-y_min)/2

    x_ellipse = []
    y_ellipse = []

    # generate coordinates for ellipse
    for i in range(360):
        x_ellipse.append(w2 * math.sin(math.radians(i)) + w2 + x_min)
        y_ellipse.append(h2 * math.cos(math.radians(i)) + h2 + y_min)

    # combine X and Y coords
    points = list(zip(x_ellipse,y_ellipse))


    # rotate points by the angle with respect to the origin
    # points_rotated = rotate(points, origin=(w2 + x_min, h2 + y_min), degrees=angle)

  
    # rotate points by the angle with respect to the midpoint
    angle_rad = np.deg2rad(angle)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    o = np.atleast_2d((w2 + x_min, h2 + y_min))
    p = np.atleast_2d(points)
    points_rotated = np.squeeze((R @ (p.T-o.T) + o.T).T)

    x_ellipse_rotated = [p[0] for p in points_rotated]
    y_ellipse_rotated = [p[1] for p in points_rotated]

    # draw original bounding box
    box1 = mpatches.Rectangle((x_min,y_min),x_max-x_min, y_max-y_min, 
                        fill = False,
                        color = "purple",
                        linewidth = 2)
    plt.gca().add_patch(box1)

    # draw rotated bounding box
    box2 = mpatches.Rectangle((min(x_ellipse_rotated),min(y_ellipse_rotated)),max(x_ellipse_rotated)-min(x_ellipse_rotated),max(y_ellipse_rotated)-min(y_ellipse_rotated), 
                        fill = False,
                        color = "red",
                        linewidth = 2)
    plt.gca().add_patch(box2)
    
    # plot ellipse
    plt.scatter(x_ellipse, y_ellipse)
    plt.scatter(x_ellipse_rotated, y_ellipse_rotated)

    plt.show()


def main():
    rotate_bounding_box(bbox=[10,10,23,20],angle=45)

if __name__ == "__main__":
    main()
 
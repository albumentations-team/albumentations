import numpy as np
import math
e = math.e

def S(a,sigma):
    FinalS = (1 + e ** (-32 * (a - sigma))) ** -1
    return FinalS

def Scene_depth(d_R,d_D,d_B,img,AtomsphericLight):
    avg_BL = np.mean(AtomsphericLight)
    avg_Ir = np.mean(img[:,:,2])
    # print('avg_BL',avg_BL)
    # print('avg_Ir',avg_Ir)

    Theta_a = S(avg_BL, 0.5*255)
    Theta_b = S(avg_Ir, 0.1*255)

    print('Theta_a',Theta_a)
    print('Theta_b',Theta_b)

    Depth_map =   Theta_b *  (Theta_a * d_D  +  (1  - Theta_a) *  d_R )  +   (1 - Theta_b) *   d_B
    return Depth_map



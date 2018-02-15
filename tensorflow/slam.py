#! /usr/bin/env python3

import numpy as np
import math
from PIL import Image
import random
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

iden = np.zeros((4,4))
iden[0,0] = 1.0
iden[1,1] = 1.0
iden[2,2] = 1.0
iden[3,3] = 1.0
iden3 = iden[0:3,0:3]
se3_x = np.zeros((4,4))
se3_x[0,3] = 1.0
se3_y = np.zeros((4,4))
se3_y[1,3] = 1.0
se3_z = np.zeros((4,4))
se3_z[2,3] = 1.0
se3_xcos = np.zeros((4,4))
se3_xcos[1,1] = 1.0
se3_xcos[2,2] = 1.0
se3_xsin = np.zeros((4,4))
se3_xsin[1,2] = -1.0
se3_xsin[2,1] = 1.0
se3_ycos = np.zeros((4,4))
se3_ycos[0,0] = 1.0
se3_ycos[2,2] = 1.0
se3_ysin = np.zeros((4,4))
se3_ysin[0,2] = 1.0
se3_ysin[2,0] = -1.0
se3_zcos = np.zeros((4,4))
se3_zcos[0,0] = 1.0
se3_zcos[1,1] = 1.0
se3_zsin = np.zeros((4,4))
se3_zsin[0,1] = -1.0
se3_zsin[1,0] = 1.0

# Lie algebra to Lie group conversion of se(3) using Rodriguez formula for so(3)

def se3exp(xi):
    B = xi[3] * se3_xsin + xi[4] * se3_ysin + xi[5] * se3_zsin
    theta = np.linalg.norm(xi[3:])
    if theta == 0:
        return iden + xi[0] * se3_x + xi[1] * se3_y + xi[2] * se3_z
    return (iden + xi[0] * se3_x + xi[1] * se3_y + xi[2] * se3_z + 
            B * math.sin(theta) / theta + 
            (B @ B) * (1.0 - math.cos(theta)) / theta / theta)

def se3log(M):
    R = M[0:3,0:3]
    ct = ((R.trace() - 1.0) / 2.0)
    theta = math.acos((R.trace() - 1.0) / 2.0)
    if ct < -0.7: # more numerically stable method when \theta is near \pi
        RmI = R - iden3
        BS = (RmI + RmI.transpose()) * (theta * theta / 4.0 / (1-ct))
        ab = BS[2,2]
        bc = BS[0,0]
        ca = BS[1,1]
        return np.array([M[0,3],M[1,3],M[2,3],
            math.sqrt(bc-ab-ca),math.sqrt(ca-bc-ab),math.sqrt(ab-ca-bc)])
    else:
        st = math.sqrt(1.0 - ct * ct)
        mult = (0.5 if st < 0.00000001 else (theta / st / 2.0))
        B = (R - R.transpose()) * mult
        return np.array([M[0,3],M[1,3],M[2,3],B[2,1],B[0,2],B[1,0]])

def imgval(I, p):
    x = int(p[0]/p[2])
    y = int(p[0]/p[2])
    w,h = I.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0.0
    return I[x,y]

def imggrad(I, p, delta):
    x = int(p[0] / p[2]);
    y = int(p[1] / p[2]);
    #print("delta")
    #print(delta)
    w,h = I.shape
    if x <= 0 or x >= w or y <= 0 or y >= w:
        return 0.0
    d0 = (delta[0] - (x - w/2) * delta[2]) / p[2]
    d1 = (delta[1] - (y - h/2) * delta[1]) / p[2]
    if math.fabs(d0) < 0.00001 and math.fabs(d1) < 0.00001:
        return 0.0
    t = math.atan2(d1, d0)
    #print(t)
    #dlsq = (d0 * d0 + d1 * d1) / (p[2] * p[2])
    dlsq = d0 * d0 + d1 * d1
    if t < 0:
        if t < -math.pi / 2:
            if t < -math.pi * 3 / 4:
                u = d1 / d0
                #print(t)
                return ((1.0 - u) * I[x-1,y] + u * I[x-1,y-1])/math.sqrt((1+u*u)*dlsq)
            else:
                u = d0 / d1
                #print(u)
                return ((1.0 - u) * I[x,y-1] + u * I[x-1,y-1])/math.sqrt((1+u*u)*dlsq)
        else:
            if t < -math.pi / 4:
                u = d0 / d1
                #print(u)
                return ((1.0 + u) * I[x,y-1] - u * I[x+1,y-1])/math.sqrt((1+u*u)*dlsq)
            else:
                u = d1 / d0
                #print(t)
                return ((1.0 + u) * I[x+1,y] - u * I[x+1,y-1])/math.sqrt((1+u*u)*dlsq)
    else:
        if t < math.pi / 2:
            if t < math.pi / 4:
                #print(t)
                u = d1 / d0
                #print(u)
                return ((1.0 - u) * I[x+1,y] + u * I[x+1,y+1])/math.sqrt((1+u*u)*dlsq)
            else:
                u = d0 / d1
                #print(u)
                return ((1.0 - u) * I[x,y+1] + u * I[x+1,y+1])/math.sqrt((1+u*u)*dlsq)
        else:
            if t < math.pi * 3 / 4:
                u = d0 / d1
                #print(u)
                return ((1.0 + u) * I[x,y+1] - u * I[x-1,y+1])/math.sqrt((1+u*u)*dlsq)
            else:
                u = d1 / d0
                #print(t)
                return ((1.0 + u) * I[x-1,y] - u * I[x-1,y+2])/math.sqrt((1+u*u)*dlsq)

DBG_image_file0=None
DBG_image_file2=None
DBG_use_mpl = 0


def slamStepIterate(Ii, Ij, K, xi, maxIter = 0):
    if maxIter <= 0:
        return xi

    xim = se3exp(xi) # matrix form of transformation

    k,three = K.shape # number of keyframes
    J = np.zeros((k, 6)) # Jacobian
    r = np.zeros((k)) # Residual vector

    p = np.array([K[:,0]*K[:,2],K[:,1]*K[:,2],K[:,2],np.ones(k)]) # screen points of keyframes
    m = xim @ p # transformed screen points of keyframes

    # calculate Jacobian
    for j,g in zip(range(6),[se3_x,se3_y,se3_z,se3_xsin,se3_ysin,se3_zsin]):
        gm = g @ m
        for i in range(k):
            J[i,j] = imggrad(Ij, m[:,i], gm[:,i])

    # calculate residuals

    #print(p.shape)
    for i in range(k):
        r[i] = imgval(Ii, p[:,i]) - imgval(Ij, m[:,i])

    # perform a step of Gauss-Newton
    JT = J.transpose()
    print("STEP ")
    try:
        dxi = np.linalg.inv(JT @ J) @ JT @ r * -1.0
    except (np.linalg.linalg.LinAlgError):
        return xi
    #print(K.shape)
    #print(m.shape)
    KK = np.array([]).transpose()
    if DBG_use_mpl==1:
        plt.imshow(mpimg.imread(DBG_image_file0))
        plt.scatter(K[0], K[1])
        plt.show()
        plt.imshow(mpimg.imread(DBG_image_file2))
        plt.scatter()
    return slamStepIterate(Ii, Ij, K, se3log(se3exp(dxi) @ xim), maxIter - 1)



    
    
    


def slamStep(pix0, dm0, pix1 = 0):
    #print(pix0.shape)
    #print(dm0.shape)
    h,w = pix0.shape
    xi0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # TODO find keyframes using some sophisticated method
    k=1000
    K=np.zeros((k,3))
    for i in range(k):
        K[i,0] = int(random.randrange(int(h * 0.1), int(h * 0.9)))
        K[i,1] = int(random.randrange(int(w * 0.1), int(w * 0.9)))
        #print(K[i,0])
        #print(K[i,1])
        K[i,2] = dm0[int(K[i,0]),int(K[i,1])]

    return slamStepIterate(pix0, pix1, K, xi0, 100)



def slamStepIf(image0, depth0, image1 = 0):
    depth0 = depth0[0,:,:,0]
    height,width = depth0.shape
    #print(depth0.shape)
    img = Image.open(image0)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.mean(img, axis=2)
    pixels0 = np.array(img).astype('float32')
    img = Image.open(image1)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.mean(img, axis=2)
    pixels1 = np.array(img).astype('float32')

    return slamStep(pixels0, depth0, pixels1)




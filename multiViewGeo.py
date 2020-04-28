import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import timeit
import cProfile
from numba import jit


# This code implements several Multiple View Geometry algorithms from Hartley and Zisserman,
# including a two-view metric reconstruction algorithm and the sparse
# Levenberg-Marquardt algorithm for bundle adjustment.
# To demonstrate the algorithm, we generate a synthetic point cloud and several camera matrices,
# and project the point cloud into each image (using the corresponding camera matrices).
# Gaussian noise is added to the image coordinates.
# We then reconstruct the point cloud using two of these images.
# The camera calibration matrices are assumed to be known,
# so we are able to obtain a metric reconstruction.
# The two-view reconstruction algorithm works as follows:
# Given a list of point correspondences between two images,
# we compute the essential matrix E for this pair of images,
# then extract camera matrices from the essential matrix E.
# We triangulate 3D point locations using the DLT method
# given in section 12.2 of Hartley and Zisserman.
# Finally, we perform bundle adjustment using the sparse Levenberg-Marquardt algorithm.

def normalizeImageCoords(imgCoords):
    
    '''
    Input: imgCoords is a 2 by numPoints numpy array.
    '''
    
    numPoints = imgCoords.shape[1]
    
    mu = np.mean(imgCoords, axis = 1)
    d = np.sqrt((imgCoords[0] - mu[0])**2 + (imgCoords[1] - mu[1])**2)
    
    scaleFactor = np.sqrt(2)/np.mean(d)
    
    T = np.array([[scaleFactor, 0, -scaleFactor*mu[0]],[0,scaleFactor, -scaleFactor*mu[1]], [0,0,1]])
    
    allOnes = np.ones((1,numPoints))
    imgCoords = np.vstack((imgCoords,allOnes))
    
    imgCoords = T @ imgCoords
    
    return imgCoords, T

def normalizePointCloudCoords(X):
    
    '''
    Input: Xis a 3 by numPoints numpy array.
    '''
    
    numPoints = X.shape[1]
    
    mu = np.mean(X, axis = 1)
    d = np.sqrt((X[0] - mu[0])**2 + (X[1] - mu[1])**2 + (X[2] - mu[2])**2)
    
    scaleFactor = np.sqrt(3)/np.mean(d)
    
    T = np.array([[scaleFactor, 0, 0, -scaleFactor*mu[0]],[0,scaleFactor, 0, -scaleFactor*mu[1]], \
                  [0, 0, scaleFactor, - scaleFactor*mu[2]], [0,0,0,1]])
    
    allOnes = np.ones((1,numPoints))
    X = np.vstack((X,allOnes))
    
    X = T @ X
    
    return X, T


def fundMatrix(img0Coords, img1Coords, normalize = True): 
    
    ''' 
    We compute the fundamental matrix F using the normalized 8 point algorithm
    given in section 11.2 of Hartley and Zisserman.
    
    Inputs: img0Coords and img1Coords are both 2 by numPoints numpy arrays.
    They are lists of corresponding points between two images of a scene.
    
    It's interesting to see how the reconstruction quality degrades
    when normalize is set to False.
    '''
    
    numPoints = img0Coords.shape[1]
    mtrx = np.zeros((numPoints,9))
    
    if normalize:
        img0Coords, T0 = normalizeImageCoords(img0Coords)
        img1Coords, T1 = normalizeImageCoords(img1Coords)
    else:
        T0 = np.eye(3) 
        T1 = np.eye(3)
    
    for j in range(numPoints):
        
        vec0 = img0Coords[:,j]
        vec1 = img1Coords[:,j]
        
        mtrx[j] = np.hstack((vec1[0]*vec0, vec1[1]*vec0,vec1[2]*vec0))
    
    U, S, VT = np.linalg.svd(mtrx, full_matrices = False)
    
    V = VT.T
    nullVec = V[:,-1]
    F = np.reshape(nullVec,(3,3))
    
    # Now enforce the rank two condition.
    U, S, VT = np.linalg.svd(F)
    S[2] = 0
    F= U @ np.diag(S) @ VT
    
    # Now denormalize.
    F = T1.T @ F @ T0
    
    ####################################################
    ## Check that the fundamental matrix is correct ####
    ####################################################
    # for j in range(numPoints):
        
    #     x0 = img0Coords[:,j]
    #     x1 = img1Coords[:,j]
    #     check = np.vdot(x1, F @ x0)
    #     print(check) # Without noise, this should be 0.
    ####################################
    
    return F


def essentialMatrix(img0Coords,img1Coords,K0,K1):
    '''
    We compute the essential matrix relating two images using
    the method given in section 9.6 of Hartley and Zisserman.
    
    Inputs: K0 and K1 are 3 by 3 numpy arrays. 
    They are the calibration matrices for cameras 0 and 1, respectively.
    img0Coords and img1Coords are 2 by numPoints numpy arrays.
    They are lists of corresponding points in the two images.
    '''
    
    F = fundMatrix(img0Coords,img1Coords, normalize = True) 
    E = K1.T @ F @ K0 # Eq. 9.12 in Hartley and Zisserman.
    
    ####################################################
    ## Check that the essential matrix E is correct ####
    ####################################################
    # numPoints = img0Coords.shape[1]
    # allOnes = np.ones((1,numPoints))
    # img0Coords = np.vstack((img0Coords,allOnes))
    # img1Coords = np.vstack((img1Coords,allOnes))
    # K0inv = np.linalg.inv(K0)
    # K1inv = np.linalg.inv(K1)
    # for j in range(numPoints):
        
    #     x0 = K0inv @ img0Coords[:,j]
    #     x1 = K1inv @ img1Coords[:,j]
    #     check = np.vdot(x1, E @ x0)
    #     print(check) # Without noise, this should be 0.
    ####################################
    
    return E


def camMatricesFromEssentialMatrix(E,K0,K1, point0, point1):
    '''
    We extract a pair of camera matrices from an essential matrix E using the
    method given in section 9.6.2 of Hartley and Zisserman.
    
    Inputs: E is a 3 by 3 numpy array (an essential matrix relating two images of a scene).
    K0 and K1 are 3 by 3 numpy arrays (calibration matrices for the corresponding cameras).
    point0 and point1 are numpy arrays with two components (a pair of corresponding points in the two images).
    '''
    
    point0 = np.linalg.inv(K0) @ np.append(point0,1)
    point1 = np.linalg.inv(K1) @ np.append(point1,1)
    
    P0 = np.hstack((np.eye(3),np.zeros((3,1)))) 
    
    U, S, VT = np.linalg.svd(E)
    # NOTE: The algorithm in section 9.6.2 requires U and V to be rotations
    # (that, is det(U) = dev(V) = 1).
    # Must be very careful about this part. Easy to forget.
    # Since E is a homogeneous matrix, we can just multiply each of U and V by -1 if necessary.
    if np.linalg.det(U) < 0: 
        U = -U
    if np.linalg.det(VT) < 0:
        VT = -VT
    
    u3 = U[:,2]
    u3 = np.reshape(u3,(3,1))
    
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    
    M = np.zeros((4,3,4))
    
    M[0] = np.hstack((U @ W @ VT,u3))
    M[1] = np.hstack((U @ W @ VT, -u3))
    M[2] = np.hstack((U @ W.T @ VT, u3))
    M[3] = np.hstack((U @ W.T @ VT, -u3))
    
    # We have four candidates for P2.
    # Now check for which candidate our reconstructed point
    # is in front of both cameras.
    # See section 9.6.3 in Hartley and Zisserman,
    # particularly figure 9.12.

    flags = np.ones(4, dtype=bool)
    
    for m in range(4):
        
        X0 = triangulatePointCloud_twoViews(P0,M[m],point0[0:2].reshape((2,1)),point1[0:2].reshape((2,1)), normalize = False)
        X0 = np.append(X0,1)
        X1 = M[m] @ X0
        
        flags[m] = (X0[2] > 0) and (X1[2] > 0)
        
    # print(flags) # Should have three False and one True.
    m = np.argmax(flags)
    return (K0 @ P0, K1 @ M[m])


def camMatricesFromPointPairs(image0Coords, image1Coords, K0, K1):  
    '''
    Inputs:  K0 and K1 are 3 by 3 numpy arrays. 
    They are the calibration matrices for cameras 0 and 1, respectively.
    image0Coords and image1Coords are 2 by numPoints numpy arrays.
    They are lists of corresponding points in the two views.
    '''

    E = essentialMatrix(image0Coords,image1Coords,K0,K1)
    
    numPoints = image0Coords.shape[1]
    j = round(numPoints/2)
    point0 = image0Coords[:,j]
    point1 = image1Coords[:,j]
    (P0,P1) = camMatricesFromEssentialMatrix(E,K0,K1, point0, point1)
    
    return (P0, P1)

def camCenterFromCamMatrix(P): 
    '''
    Input: P is a 3 by 4 numpy array (a camera matrix).
    '''
    
    U, S, VT = np.linalg.svd(P)
    V = VT.T
    nullVec = V[:,3]
    
    return (nullVec/nullVec[3])[0:3]


def triangulatePointCloud_twoViews(P0,P1,img0Coords,img1Coords, normalize = True):
    '''
    Given camera matrices for two views of a scene,
    and a list of corresponding points in the two views,
    we reconstruct a point cloud using the DLT method given in
    section 12.2 of Hartley and Zisserman.
    
    Inputs: P0 and P1 are 3 by 4 numpy arrays (camera matrices).
    img0Coords and img1Coords are 2 by numPoints numpy arrays.
    '''
    
    numPoints = img0Coords.shape[1]
    
    # First normalize image coordinates and adjust P0 and P1 accordingly.
    if normalize:
        img0Coords, T0 = normalizeImageCoords(img0Coords)
        img1Coords, T1 = normalizeImageCoords(img1Coords)
    else:
        T0 = np.eye(3)
        T1 = np.eye(3)
    
    P0 = T0 @ P0
    P1 = T1 @ P1

    
    X_est = np.zeros((3,numPoints))
    
    for j in range(numPoints):
        
        x0, y0 = img0Coords[0:2,j]
        x1, y1 = img1Coords[0:2,j]
        
        A0 = np.vstack((x0*P0[2] - P0[0], y0*P0[2] - P0[1]))
        A1 = np.vstack((x1*P1[2] - P1[0], y1*P1[2] - P1[1]))
        A = np.vstack((A0,A1))
        
        U, S, VT = np.linalg.svd(A)    
        V = VT.T
        X0 = V[:,3]
        X0 = X0/X0[3]
        
        X_est[:,j] = X0[0:3]
        
    return X_est


def projectPointCloudIntoImages(X, camMatrices):
    '''
    Inputs: X is 3 by numPoints.
    camMatrices is numCameras by 3 by 4.
    '''
                            
    numCameras = camMatrices.shape[0]    
    numPoints = X.shape[1]
    points = np.vstack((X, np.ones(numPoints)))
    
    imageCoords = np.zeros((numCameras,2,numPoints))
    
    for i in range(numCameras):
        
        p = camMatrices[i] @ points
        
        xCoords = p[0]/p[2]
        yCoords = p[1]/p[2]
        
        imageCoords[i] = np.vstack((xCoords,yCoords))
        
    return imageCoords

def camFromPointProjections(X, imgCoords, normalize = True):
    '''
    Inputs: X is 3 by numPoints.
    imgCoords is 2 by numPoints.
    This function finds a camera matrix P such that projecting the points in X
    yields the points in imgCoords.
    It's interesting to see how, when noise is added to the image
    coordinates, the result without normalization is terrible.
    '''
    X_orig = X.copy()
    imgCoords_orig = imgCoords.copy()  
    numPoints = X.shape[1]

    if normalize:
        X, T_space = normalizePointCloudCoords(X)
        imgCoords, T_image = normalizeImageCoords(imgCoords)
    else:
        allOnes = np.ones((1,numPoints))
        X = np.vstack((X,allOnes))
        imgCoords = np.vstack((imgCoords,allOnes))
        T_space = np.eye(4)
        T_image = np.eye(3)
    
    # allOnes = np.ones((1,numPoints))
    # X = np.vstack((X,allOnes))
    
    A = np.zeros((2*numPoints, 12))
    
    for j in range(numPoints):
        
        Xj = X[:,j]
        xj, yj = imgCoords[0:2,j]
        A[2*j] = np.hstack((np.zeros(4), -Xj, yj*Xj))
        A[2*j+1] = np.hstack((Xj,np.zeros(4),-xj*Xj))
        
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    P = V[:,-1]
    P = np.reshape(P,(3,4))
    
    P = np.linalg.inv(T_image) @ P @ T_space
    
    ### Now check that P is correct. ###
    # allOnes = np.ones((1,numPoints))
    # X_orig = np.vstack((X_orig,allOnes))
    # PX = P @ X_orig
    # xCoords = PX[0]/PX[2]
    # yCoords = PX[1]/PX[2]
    # imgCoordsCheck = np.vstack((xCoords,yCoords)) # DEBUG THIS. NORMALIZATION IS NOT CORRECT YET.

    # diff = np.max(np.abs(imgCoords_orig - imgCoordsCheck)) # Without noise this should be zero.
    # for j in range(numPoints):
    #     print(imgCoords_orig[:,j], imgCoordsCheck[:,j]) # Without noise these should agree.
        
    return P

# @jit(nopython = True)
def xHat(P,X):
    '''
    Inputs: P is a 3 by 4 numpy array (a camera matrix).
    X is a numpy array with three components (a point in space).
    
    this function takes as input a camera P and point X in space
    and returns the image of X (which is a point with two components).
    '''
    
    out = (P[0:2,0:3] @ X + P[0:2,3])/(np.vdot(P[2,0:3], X) + P[2,3])
    
    ###
    # X_aug = np.append(X,1)
    
    # out = np.zeros(2)
    # denom = np.vdot(P[2],X_aug)
    # out[0] = np.vdot(P[0],X_aug)/denom
    # out[1] = np.vdot(P[1], X_aug)/denom
    
    return out

# @jit(nopython = True)
def dxHatdX(P,X):
    '''
    Inputs: P is a 3 by 4 numpy array (a camera matrix).
    X is a numpy array with three components (a point in space).
    
    Here xHat is a function that takes as input a camera P and point X in space
    and returns the image of X (which is a point with two components).
    '''
    
    X_aug = np.append(X,1)
    out = np.zeros((2,3))
    
    out[0] = (np.vdot(P[2],X_aug)*P[0,0:3] - np.vdot(P[0],X_aug)*P[2,0:3])/np.vdot(P[2],X_aug)**2
    
    out[1] = (np.vdot(P[2], X_aug)*P[1,0:3] - np.vdot(P[1], X_aug)*P[2,0:3])/np.vdot(P[2],X_aug)**2
    
    return out

def dxHatdX_est(P,X):
    
    h = 1e-6
    
    out = np.zeros((2,3))
    

    for k in range(3):
        
        X_plus = X.copy()
        X_plus[k] += h
        
        X_minus = X.copy()
        X_minus[k] -= h
        
        out[:,k] = (xHat(P,X_plus) - xHat(P,X_minus))/(2*h)
        
    return out
    

# @jit(nopython = True)
def dxHatdP(P,X):
    '''
    Inputs: P is a 3 by 4 numpy array (a camera matrix).
    X is a numpy array with three components (a point in space).
    
    Here xHat is a function that takes as input a camera P and point X in space
    and returns the image of X (which is a point with two components).
    '''
    
    X_aug = np.append(X,1)
    
    out = np.zeros((2,12))
    out[0,0:4] = X_aug/np.vdot(P[2],X_aug)
    out[0,4:8] = np.zeros(4)
    out[0,8:] = (-np.vdot(P[0],X_aug)/np.vdot(P[2],X_aug)**2)*X_aug
    
    out[1,0:4] = np.zeros(4)
    out[1,4:8] = X_aug/np.vdot(P[2],X_aug)
    out[1,8:] = (-np.vdot(P[1],X_aug)/np.vdot(P[2],X_aug)**2)*X_aug
    
    return out

# @jit(nopython = True)
def dxHatdP_est(P,X):
    
    h = 1e-6
    out = np.zeros((2,12))
    
    idx = 0  
    for i in range(3):
        for j in range(4):
        
            P_plus = P.copy()
            P_plus[i,j] += h
            
            P_minus = P.copy()
            P_minus[i,j] -= h
            
            out[:,idx] = (xHat(P_plus,X) - xHat(P_minus,X))/(2*h)
            
            idx += 1
            
    return out

# @jit(nopython = True)
def reprojectionError_notVectorized(camMatrices, X_est, imgCoords):
    
    numCameras = camMatrices.shape[0]
    numPoints = X_est.shape[1]
    
    cost = 0
    
    for i in range(numCameras):
        
        Pi = camMatrices[i]
        
        for j in range(numPoints):
            
            Xj = X_est[:,j]
            x_ij = imgCoords[i,:,j]
            cost += np.linalg.norm(x_ij - xHat(Pi,Xj))**2
            
    return cost

# @jit(nopython = True)
def reprojectionError(camMatrices, X_est, imgCoords):
    '''
    Inputs: camMatrices is a numCameras by 3 by 4 numpy array.
    X_est is a 3 by numPoints numpy array.
    imgCoords is a 2 by numPoints numpy array.
    '''
    
    numCameras = camMatrices.shape[0]
    numPoints = X_est.shape[1]
    
    proj = camMatrices @ np.vstack((X_est,np.ones((1,numPoints))))
    cost = np.sum((proj[:,0,:]/proj[:,2,:] - imgCoords[:,0,:])**2 + (proj[:,1,:]/proj[:,2,:] - imgCoords[:,1,:])**2)
            
    return cost
            

# @jit(nopython = True)
def bundleAdjust_LM(camMatrices, X_est, imgCoords):
    '''
    We perform bundle adjustment using the Levenberg-Marquardt algorithm.
    See bundleAdjust_sparseLM for the sparse version of this algortihm.
    
    Inputs: camMatrices is a numCameras by 3 by 4 numpy array.
    X_est is a 3 by numPoints numpy array.
    imgCoords is a numCameras by 2 by numPoints numpy array.
    '''
    
    lmbda = .0001
    maxIter = 10
    reduceFactor = .1
    increaseFactor = 10
    
    numCameras = camMatrices.shape[0]
    numPoints = X_est.shape[1]
    
    d = 12*numCameras + 3*numPoints
    
    P = camMatrices.copy()
    X = X_est.copy()
    
    reprojErr = reprojectionError(P,X,imgCoords)
    reprojErrors = np.zeros(maxIter)
    
    for iter in range(maxIter):
        
        print('iter is: ', iter)
        
        # Set up the normal equations.
        mtrx = np.zeros((d,d))
        rhs = np.zeros(d)
        
        for i in range(numCameras):
            
            Pi = P[i]
            rows = np.zeros((12, d))
            
            for j in range(numPoints):
                
                Xj = X[:,j]
                x_ij = imgCoords[i,:,j]
                
                dxHatdP_ij = dxHatdP(Pi,Xj)
                dxHatdX_ij = dxHatdX(Pi,Xj)
                
                rows[:,12*i:12*i+12] += dxHatdP_ij.T @ dxHatdP_ij
                rows[:,12*numCameras + 3*j:12*numCameras + 3*j + 3] = dxHatdP_ij.T @ dxHatdX_ij
                
                rhs[12*i:12*i+12] += dxHatdP_ij.T @ (x_ij - xHat(Pi,Xj))
                
            
            rows[:,12*i:12*i+12] += lmbda * np.eye(12)
            
            mtrx[12*i:12*i+12,:] = rows
            
        for j in range(numPoints):
            
            Xj = X[:,j]
            rows = np.zeros((3,d))
            
            for i in range(numCameras):
                
                Pi = P[i]
                x_ij = imgCoords[i,:,j]
                dxHatdP_ij = dxHatdP(Pi,Xj)
                dxHatdX_ij = dxHatdX(Pi,Xj)
                
                rows[:,12*i:12*i+12] = dxHatdX_ij.T @ dxHatdP_ij
                rows[:,12*numCameras + 3*j:12*numCameras + 3*j+3] += dxHatdX_ij.T @ dxHatdX_ij
                
                rhs[12*numCameras + 3*j:12*numCameras + 3*j + 3] += dxHatdX_ij.T @ (x_ij - xHat(Pi,Xj))
                
            rows[:,12*numCameras + 3*j:12*numCameras + 3*j+3] += lmbda * np.eye(3)
            
            mtrx[12*numCameras + 3*j:12*numCameras + 3*j + 3] = rows
            
        # Now solve normal equations
        vec = np.linalg.solve(mtrx, rhs)
        
        dP = vec[0:12*numCameras]
        dX = vec[12*numCameras:]
        
        dP = np.reshape(dP,(numCameras,3,4))
        dX = np.reshape(dX,(numPoints,3)).T

        P_next = P + dP
        X_next = X + dX
        
        reprojErr_next = reprojectionError(P_next, X_next, imgCoords)
        
        if reprojErr_next < reprojErr:
            
            P = P_next
            X = X_next
            reprojErr = reprojErr_next
            lmbda *= reduceFactor

        else:
            lmbda *= increaseFactor
            reprojErrors[iter] = reprojErr
            
        reprojErrors[iter] = reprojErr
        print('iter: ', iter, 'reprojError: ', reprojErr_next, 'lmbda: ', lmbda)
        
    return P, X, reprojErrors


# @jit(nopython = True)
def bundleAdjust_sparseLM(camMatrices, X_est, imgCoords):
    '''
    We perform bundle adjustment using the sparse Levenberg-Marquardt algorithm.
    
    Inputs: camMatrices is a numCameras by 3 by 4 numpy array.
    X_est is a 3 by numPoints numpy array.
    imgCoords is a numCameras by 2 by numPoints numpy array.
    '''
    
    lmbda = .0001
    maxIter = 10
    reduceFactor = .1
    increaseFactor = 10
    
    numCameras = camMatrices.shape[0]
    numPoints = X_est.shape[1]
    
    d = 12*numCameras + 3*numPoints
    
    P = camMatrices.copy()
    X = X_est.copy()
    
    reprojErr = reprojectionError(P,X,imgCoords)
    reprojErrors = np.zeros(maxIter)
    
    for iter in range(maxIter):
        
        print('iter is: ', iter)
        
        # Set up the normal equations.
        
        # upperLeftBlocks = np.zeros((numCameras,12,12))
        # coefficient matrix for normal equations is partitioned into
        # upper left (UL), upper right (UR), lower left (LL), and lower right (LR) blocks.
        # Lower left block is transpose of upper right block.
        # The lower right matrix is itself block diagonal.
        UL = np.zeros((12*numCameras,12*numCameras))
        UR = np.zeros((12*numCameras,3*numPoints))
        LR_blocks= np.zeros((numPoints,3,3))
        rhs = np.zeros(d) # right hand side of normal equations.
        
        for i in range(numCameras):
            
            Pi = P[i]
            rows = np.zeros((12, d))
            
            for j in range(numPoints):
                
                Xj = X[:,j]
                x_ij = imgCoords[i,:,j]
                
                dxHatdP_ij = dxHatdP(Pi,Xj)
                dxHatdX_ij = dxHatdX(Pi,Xj)
                
                rows[:,12*i:12*i+12] += dxHatdP_ij.T @ dxHatdP_ij
                rows[:,12*numCameras + 3*j:12*numCameras + 3*j + 3] = dxHatdP_ij.T @ dxHatdX_ij
                
                UL[12*i:12*i+12,12*i:12*i+12] += dxHatdP_ij.T @ dxHatdP_ij
                UR[12*i:12*i+12,3*j:3*j + 3] = dxHatdP_ij.T @ dxHatdX_ij
                
                rhs[12*i:12*i+12] += dxHatdP_ij.T @ (x_ij - xHat(Pi,Xj))
                
            UL[12*i:12*i+12,12*i:12*i+12] += lmbda * np.eye(12)
            
        for j in range(numPoints):
            
            Xj = X[:,j]
            
            for i in range(numCameras):
                
                Pi = P[i]
                x_ij = imgCoords[i,:,j]
                dxHatdP_ij = dxHatdP(Pi,Xj)
                dxHatdX_ij = dxHatdX(Pi,Xj)
                
                LR_blocks[j] += dxHatdX_ij.T @ dxHatdX_ij
                
                rhs[12*numCameras + 3*j:12*numCameras + 3*j + 3] += dxHatdX_ij.T @ (x_ij - xHat(Pi,Xj))
                
            LR_blocks[j] += lmbda * np.eye(3)
            
        rhs_upper = rhs[0:12*numCameras]
        rhs_lower = rhs[12*numCameras:]
        
        LR_blocks_inv = np.zeros(LR_blocks.shape)
        LR_inv_times_rhs_lower = np.zeros(rhs_lower.shape)
        LR_inv_times_LL = np.zeros(UR.T.shape)
        for j in range(numPoints):
            LR_blocks_inv[j] = np.linalg.inv(LR_blocks[j])
            LR_inv_times_rhs_lower[3*j:3*j+3] = LR_blocks_inv[j] @ rhs_lower[3*j:3*j+3]
            LR_inv_times_LL[3*j:3*j+3] = LR_blocks_inv[j] @ UR.T[3*j:3*j+3,:]
            
        dP = np.linalg.solve(UL- UR @ LR_inv_times_LL, rhs_upper - UR @ LR_inv_times_rhs_lower)
        dX = LR_inv_times_rhs_lower - LR_inv_times_LL @ dP
        
        dP = np.reshape(dP,(numCameras,3,4))
        dX = np.reshape(dX,(numPoints,3)).T

        P_next = P + dP
        X_next = X + dX
        
        reprojErr_next = reprojectionError(P_next, X_next, imgCoords)
        
        if reprojErr_next < reprojErr:
            
            P = P_next
            X = X_next
            reprojErr = reprojErr_next
            lmbda *= reduceFactor

        else:
            lmbda *= increaseFactor
            reprojErrors[iter] = reprojErr
            
        reprojErrors[iter] = reprojErr
        print('iter: ', iter, 'reprojError: ', reprojErr_next, 'lmbda: ', lmbda)
        
    return P, X, reprojErrors
                

def createFakePointCloud():
    '''
    This function creates a simple point cloud that can be used
    to test multiview reconstruction algorithms.
    '''
    
    X = np.zeros((3,4,9,3))
    
    numRows = 3
    numCols = 5
    depth = 7
    
    X = np.zeros((3,numRows*numCols*depth))
    
    j = 0
    for q in range(numRows):
        for r in range(numCols):
            for s in range(depth):
                
                X[:,j] = np.array([q-2,r-2,s])
                j += 1

    return X


def createFakeCameras():
    '''
    This function creates some camera matrices that
    can be used to test multiview reconstruction algorithms.
    
    The z-axis of the camera coordinate frame points out of the front of the camera.
    The x-axis of the camera coordinate frame points out of the top of the camera.
    The y-axis of the camera coordinate frame points out of the right side of the camera.
    
    The cameras will be evenly spaced along a circle of radius R, all pointing
    at the origin of the world coordinate system.
    '''
    
    focalLength = .03 # I'm thinking in units of meters.
    numCameras = 7
    R = 10
    deltaTheta = np.pi/numCameras
    thetaVals = [deltaTheta*i for i in range(numCameras)]
    camCenters = np.zeros((numCameras,3))
    camBases = np.zeros((numCameras,3,3)) 
    
    for i in range(numCameras):
        
        theta = thetaVals[i]
        camCenters[i] = np.array([R*np.cos(theta),R*np.sin(theta),2]) # In world coordinates.
        
        u3 = np.array([-np.cos(theta),-np.sin(theta),0]) # Camera points towards origin of world coordinate system.
        u1 = np.array([0,0,1])
        u2 = np.cross(u3,u1)
        
        camBases[i] = np.vstack((u1,u2,u3)).T
       
    # All cameras will have the same calibration matrix K.
    K = np.array([[focalLength,0,0],[0,focalLength,0],[0,0,1]])
    camMatrices = np.zeros((numCameras,3,4))
    for i in range(numCameras):
     
        C = camCenters[i]
        U = camBases[i]
        
        M = np.zeros((3,4))
        M[:,0:3] = U.T
        M[:,3] = -U.T @ C
        
        camMatrices[i] = K @ M
        
    return camMatrices, K, camCenters


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input: ax is a matplotlib axis, e.g., as output from plt.gca().
    '''
    # This function is taken from https://stackoverflow.com/a/31364297/1854748

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
 
if __name__ == '__main__':
    
    np.random.seed(19)
    
    # Create point cloud and cameras.
    X = createFakePointCloud()
    camMatrices, K, camCenters = createFakeCameras()
    
    # Project point cloud into images.
    imageCoords = projectPointCloudIntoImages(X, camMatrices)
    noise = 1*.0002645* np.random.randn(*imageCoords.shape) # 1 pixel is .0002645 meters.
    # noise = 0*noise
    imageCoords = imageCoords + noise
    
    # Visualize the point cloud and cameras.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[0], X[1], X[2], color = 'blue')
    ax.scatter3D(camCenters[:,0],camCenters[:,1],camCenters[:,2], color = 'orange')
    plt.title('Ground truth point cloud and camera centers')
    set_axes_equal(ax)
    
    # Now reconstruct cameras and point cloud from list of point pairs
    # between two views.
       
    i = 2 # We'll do a two-view reconstruction using image 0 and image i.
    
    P0, Pi = camMatricesFromPointPairs(imageCoords[0], imageCoords[i], K, K)
    X_est = triangulatePointCloud_twoViews(P0,Pi,imageCoords[0],imageCoords[i], normalize = True)
    
    center_0 = camCenterFromCamMatrix(P0)
    center_i = camCenterFromCamMatrix(Pi)
        
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(X_est[0], X_est[1], X_est[2], color = 'blue')
    # ax.scatter3D(*center_0, color = 'orange')
    # ax.scatter3D(*center_i, color = 'orange')
    # plt.title('Two view reconstruction of point cloud and camera positions (no bundle adjustment)')
    # set_axes_equal(ax)
    
    
    #############
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_est[0], X_est[1], X_est[2], color = 'blue')
    numCameras = camMatrices.shape[0]
    camMatrices_est = np.zeros((numCameras,3,4))
    for i in range(numCameras):
        
        Pi = camFromPointProjections(X_est, imageCoords[i], normalize = True)
        camMatrices_est[i] = Pi
        
        center_i = camCenterFromCamMatrix(Pi)
        ax.scatter3D(*center_i, color = 'orange')
        
    plt.title('Two-view point cloud reconstruction (before bundle adjustment)')
    set_axes_equal(ax)
    
    #%%
    
    # cProfile.run('bundleAdjust_sparseLM(camMatrices_est, X_est, imageCoords)')
    
    #%%

    timeStart = timeit.default_timer()
    P_adjusted, X_adjusted, reprojErrors = bundleAdjust_sparseLM(camMatrices_est, X_est, imageCoords)
    timeSparse = timeit.default_timer() - timeStart
    
    # timeStart = timeit.default_timer()
    # P_adjusted_check, X_adjusted_check, reprojErrors_check = bundleAdjust_LM(camMatrices_est, X_est, imageCoords)
    # timeNonSparse = timeit.default_timer() - timeStart
    
    # plt.figure()
    # plt.plot(reprojErrors)
    # plt.title('Reprojection error versus iteration')
    
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_adjusted[0], X_adjusted[1], X_adjusted[2], color = 'blue')

    for i in range(numCameras):
        
        Pi = P_adjusted[i]
        center_i = camCenterFromCamMatrix(Pi)
        ax.scatter3D(*center_i, color = 'orange')
        
    plt.title('Points and camera centers after bundle adjustment')
    set_axes_equal(ax)
    
    
    #%%
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(X_adjusted_check[0], X_adjusted_check[1], X_adjusted_check[2], color = 'blue')

    # for i in range(numCameras):
        
    #     Pi = P_adjusted_check[i]
    #     center_i = camCenterFromCamMatrix(Pi)
    #     ax.scatter3D(*center_i, color = 'orange')
        
    # plt.title('Points and camera centers after non-sparse bundle adjustment.')
    # set_axes_equal(ax)

    # print(np.max(np.abs(X_adjusted - X_adjusted_check))/np.max(np.abs(X_adjusted)))
    # print(np.max(np.abs(P_adjusted - P_adjusted_check))/np.max(np.abs(P_adjusted)))
    # print(np.max(np.abs(reprojErrors - reprojErrors_check)))







    
    




















            
            
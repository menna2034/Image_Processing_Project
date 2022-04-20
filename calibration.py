import numpy as np
import cv2
import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# ------------------------------------
# return 3D coordinates for board corners of given size
def calDistortMatrix( ifPlot=False): 
# Mapping each calibration image to number of checkerboard corners
# Everything is (9,6) for now
    objp_dict = {
    1: (9, 5),
    2: (9, 6),
    3: (9, 6),
    4: (7, 4),
    5: (7, 6),
    6: (9, 6),
    7: (9, 6),
    8: (9, 6),
    9: (9, 6),
    10: (9, 6),
    11: (9, 6),
    12: (9, 6),
    13: (9, 6),
    14: (9, 6),
    15: (9, 6),
    16: (9, 6),
    17: (9, 6),
    18: (9, 6),
    19: (9, 6),
    20: (9, 6),
    }

    # List of object points and corners for calibration
    objp_list = []
    corners_list = []
    i = 0
    # Go through all images and find corners
    for k in objp_dict:
        nx, ny = objp_dict[k]
        sizeBoardCorners= nx, ny
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # Make a list of calibration images
        fname = 'camera_cal/calibration%s.jpg' % str(k) #'camera_cal/calibration*.jpg'
        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, save & draw corners
        if ret == True:
            # Save object points and corresponding corners
            objp_list.append(objp)
            corners_list.append(corners)

            if ifPlot:
            # draw the board and the found corners to make sure all are found correctly
                cv2.drawChessboardCorners(img, sizeBoardCorners, corners, ret)
                plt.subplot(5, 5, i+1)
                plt.imshow(img)
                i += 1
        else:
            print('Warning: ret = %s for %s' % (ret, fname))
    if ifPlot:
        plt.show()


    # Calibrate camera and undistort a test image
    img = cv2.imread('test_images/straight_lines1.jpg')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, corners_list, img_size,None,None)
    return ret, mtx, dist


# ------------------------------------
# Perspective transform of a rectangular
# Selected points are hard-coded
def computePerspective(ifPlot=False):
    
    # make a copy of image for drawing
    testFn = 'test_images/straight_lines1.jpg'
    img_orig = cv2.imread(testFn)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Four corner points of the rectangular used to map perspective
    points_orig = np.array([[312,650], [1007, 650], [720, 470], [568, 470]])
    points_targ = np.array([[360,650], [850, 650], [850, 100], [360, 100]])

    # Compute the forward and reverse persepective transform matrices
    M         = cv2.getPerspectiveTransform(np.float32(points_orig), np.float32(points_targ))
    M_inv = cv2.getPerspectiveTransform(np.float32(points_targ), np.float32(points_orig))
    
    # Apply the forward transform to get bird view image
    img_bird = cv2.warpPerspective(img_orig, M, (img_orig.shape[1], img_orig.shape[0]) , flags=cv2.INTER_LINEAR)

    if ifPlot:
        # Draw boundaries of the rectangular in both views
        cv2.polylines(img_orig, np.int32([points_orig]), isClosed=True, color=[255, 0 ,0], thickness=3)
        cv2.polylines(img_bird, np.int32([points_targ]), isClosed=True, color=[0, 0, 255], thickness=3)
        
        # Display
        f, axhandles = plt.subplots(1, 2, figsize=(20,10))

        axhandles[0].imshow(img_orig)
        axhandles[0].set_title('Original Image with 4 Source points')

        axhandles[1].imshow(img_bird)
        axhandles[1].set_title('Bird-view Image with 4 Target points')
        
        plt.show()


    return M, M_inv


def computePerspective2(testFn, ifPlot=False):
    
    # make a copy of image for drawing
    # testFn = 'test_images/straight_lines1.jpg'
    # img_orig = cv2.imread(testFn)
    # img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig =testFn
    
    # Four corner points of the rectangular used to map perspective
    points_orig = np.array([[312,650], [1007, 650], [720, 470], [568, 470]])
    points_targ = np.array([[360,650], [850, 650], [850, 100], [360, 100]])

    # Compute the forward and reverse persepective transform matrices
    M         = cv2.getPerspectiveTransform(np.float32(points_orig), np.float32(points_targ))
    M_inv = cv2.getPerspectiveTransform(np.float32(points_targ), np.float32(points_orig))
    
    # Apply the forward transform to get bird view image
    img_bird = cv2.warpPerspective(img_orig, M, (img_orig.shape[1], img_orig.shape[0]) , flags=cv2.INTER_LINEAR)
    cv2.polylines(img_orig, np.int32([points_orig]), isClosed=True, color=[255, 0 ,0], thickness=3)
    cv2.polylines(img_bird, np.int32([points_targ]), isClosed=True, color=[0, 0, 255], thickness=3)
    
    if ifPlot:
        # Draw boundaries of the rectangular in both views
 
        # Display
        f, axhandles = plt.subplots(1, 2, figsize=(20,10))

        axhandles[0].imshow(img_orig)
        axhandles[0].set_title('Original Image with 4 Source points')

        axhandles[1].imshow(img_bird)
        axhandles[1].set_title('Bird-view Image with 4 Target points')
        
        plt.show()
    # if testFn=='test_images/straight_lines1.jpg':  

    return img_orig, img_bird    


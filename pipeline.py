import numpy as np
import cv2
from sklearn import pipeline

from calibration import *
from laneFinder import LaneFinder

from thresholder import *

import matplotlib.pyplot as plt
from debug_mode import *



class PipeLine:
    def __init__(self):

        self.isInitialized = False
            
    # Run len calibration, estimate view transform matrix
    def initialize(self):
        
        # Run lens calibration with all calibration images, store distortion matrix
        print('Pipeline Initialization: Calculating lens distortion ...')
        sizeBoardCorners = (9, 6)
        ret, mtx, dist = calDistortMatrix(ifPlot=False)
        self.distortionMtx = mtx
        self.distortionDist = dist
    
        # Calculate view transform matrices (forward and backward)
        print('Pipeline Initialization: Calculating bird view transform matrix ...')
        self.M, self.M_inv = computePerspective()
        
        # Initialize a LaneFinder object
        print('Pipeline Initialization: Initializing LaneFinder instance ...')
        self.laneFinderObj = LaneFinder(self.M, self.M_inv)
        
        self.isInitialized = True

        
    def run_Pipeline(self, img):
        from main import mode 
        
        if not self.isInitialized:
            self.initialize()
            

            
        # Fix lens distortion
        img_undist = cv2.undistort(img, self.distortionMtx, self.distortionDist, None, self.distortionMtx)
    
        # Generate mask for lane edges and transform to bird view
        thObj = Thresholder()
        mask = thObj.threshold(img) #combinedMask
        mask_bird = cv2.warpPerspective(np.float64(mask), self.M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
        # combinedMask , orig_img, bird_img, mask_bird, img_with_lanes = my_fun(img)
        # Run the lane finder process with includes:
        # 1. Find lane lines with sliding window;
        # 2. Fit the lane line coefficients;
        # 3. Compute curvature;
        # 4. Mark image with lane lines/regions and display info as text;
        
        
        res = self.laneFinderObj.run(img, mask_bird)
        # print(type(res))
        # Add additional plot
        out=res
        if mode == "0": ### 0 for debuging mode 
                           ### 1 for runtime mode
            out = self.overlay(res, mask, self.laneFinderObj.mask_laneAndWindow, mask_bird)
            # out = self.overlay(res)
        return out



    def overlay(self, fullimg, m1, m2, m3):
        sizr, sizc, dummy = fullimg.shape

        # Shrink size 
        m1 = m1[::4, ::4]
        m2 = m2[::4, ::4, :]
        m3 = m3[::4, ::4]
        # m4 = m4[::4, ::4, :]
        # m5 = m5[::4, ::4]

        sr, sc = m1.shape
        sr_3, sc_3 = m3.shape
        for ch in range(3):
            fullimg[0:sr, sizc-sc:sizc, ch] = m1*255
            fullimg[2*sr+20:2*sr+sr_3+20, sizc-sc:sizc, ch] = m3*255
            
        fullimg[sr+10:2*sr+10, sizc-sc:sizc, :] = m2

        
        # Add Legend Text
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(fullimg, text="Masked Lane Edge (Orig View)", org=(sizc-sc,10), fontFace=font, fontScale=0.5, color=(102,178,255), thickness=1)
        # cv2.putText(fullimg, text="Lane Detection (Bird View)", org=(sizc-sc,sr+10), fontFace=font, fontScale=0.5, color=(255,51,255), thickness=1)

        return fullimg

# p=PipeLine()
# fn = 'test_images/test3.jpg'
# img = cv2.imread(fn)
# p.run_Pipeline(fn)
     
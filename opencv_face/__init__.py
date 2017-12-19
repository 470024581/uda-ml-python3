

import cv2
import numpy as np  
# from find_obj import filter_matches,explore_match  
from matplotlib import pyplot as plt  

# help(cv2)


def getSift():
    img1 = cv2.imread('data/3.png')
    img2 = cv2.imread('data/4.png')
    gray1=cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    matches = cv2.BFMatcher().knnMatch(des1,des2,k=2)
    
    
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#     search_params = dict(checks = 50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    print(len(matches))
    
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            print(m, n)
            good.append(m)
    print(len(good))
    
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
#     cv2.draw
    plt.imshow(img3)
    plt.show()
    
#     img=cv2.drawKeypoints(image=img1,outImage=img1, keypoints=kp1)  
#     plt.imshow(img)
#     plt.show()
#     img=cv2.drawKeypoints(image=img2,outImage=img2, keypoints=kp2)  
#     plt.imshow(img)
#     plt.show()

getSift()

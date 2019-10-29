import requests
import numpy as np
import cv2
import math
import time
from array import *
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib, time

#initialize variables for plotting
cx1 = 0
cx2 = 0
cy1 = 0
cy2 = 0
cz1 = 0
cz2 = 0


# initialize the known distance from the camera to the object, which
# in this case is 15 inches
KNOWN_DISTANCE = 15.0
KNOWN_HEIGHT = 11.0

focalLength = (720 * KNOWN_DISTANCE) / KNOWN_HEIGHT

MIN_MATCH_COUNT = 30
detector = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
flannParam = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
flann = cv2.FlannBasedMatcher(flannParam,())

trainImage = cv2.imread('1.jpg', 0)
trainKP, trainDescriptors = detector.detectAndCompute(trainImage, None)
trainImage2 = cv2.imread('2.jpg',0)
tKP2, tDesc2 = detector.detectAndCompute(trainImage2, None)


# trainImage = cv2.imread('4box_new.jpeg', 0)
# trainKP, trainDescriptors = detector.detectAndCompute(trainImage, None)
# trainImage2 = cv2.imread('Three.jpg',0)
# tKP2, tDesc2 = detector.detectAndCompute(trainImage2, None)

fig = plt.figure()
ax = plt.axes(projection='3d')

cam = cv2.VideoCapture("final/near.mp4")
# cam = cv2.VideoCapture("final/angle.mp4")
# cam = cv2.VideoCapture("final/away.mov")
# cam = cv2.VideoCapture("final/light.mov")
# cam = cv2.VideoCapture("final/error.mp4")

while True:
	#scanning for the first training image
    ret, queryImageBGR = cam.read()
    queryImage = cv2.cvtColor(queryImageBGR, cv2.COLOR_BGR2GRAY)
    queryKP, queryDescriptors = detector.detectAndCompute(queryImage, None)
    matches = flann.knnMatch(queryDescriptors, trainDescriptors, 2)
    goodMatch1 = []
    goodMatch2 = []

    for m, n in matches:
        if(m.distance < 0.75*n.distance):
            goodMatch1.append(m)

    if(len(goodMatch1)>MIN_MATCH_COUNT):
        tp = []
        qp = []
        for m in goodMatch1:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp = np.float32((tp,qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImage.shape
        trainBorder = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        marker = (queryBorder[3][0][0] - queryBorder[0][0][0])

		#Finding the coordinates of the object in inches
        x0 = (queryBorder[1][0][0] - 640)/128
        y0 = (360 - queryBorder[1][0][1])/128
        x1 = (queryBorder[2][0][0] - 640)/128
        y1 = (360 - queryBorder[2][0][1])/128
        x2 = (queryBorder[0][0][0] - 640)/128
        y2 = (360 - queryBorder[0][0][1])/128
        x3 = (queryBorder[3][0][0] - 640)/128
        y3 = (360 - queryBorder[3][0][1])/128

        a = ((x1 - x0)*(y3 - y1) - (x3 - x1)*(y1 - y0))/((x3 - x1)*(y2 - y1) - (x2 - x1)*(y3 - y1))
        b = ((x1 - x0)*(y2 - y1) - (x2 - x1)*(y1 - y0))/((x3 - x1)*(y2 - y1) - (x2 - x1)*(y3 - y1))

        t0 = KNOWN_HEIGHT/math.sqrt((a*x2 - x0)*(a*x2 - x0) + (a*y2 - y0)*(a*y2 - y0) + focalLength*focalLength*(a - 1)*(a - 1)/(128*128))
        t2 = a*t0
        t3 = b*t0
        t1 = t0 + t3 - t2

        x0 = t0*x0
        x1 = t1*x1
        x2 = t2*x2
        x3 = t3*x3
        y0 = t0*y0
        y1 = t1*y1
        y2 = t2*y2
        y3 = t3*y3
        z0 = t0*focalLength/128
        z1 = t1*focalLength/128
        z2 = t2*focalLength/128
        z3 = t3*focalLength/128

        cx1 = (x1 + x2 + x3 + x0)/4
        cy1 = (y1 + y2 + y3 + y0)/4
        cz1 = (z0 + z1 + z2 + z3)/4

        dist = math.sqrt(cx1*cx1 + cy1*cy1 + cz1*cz1)

        cv2.putText(queryImageBGR, "(%.2f, %.2f, %.2f)" % (cx1, cy1, cz1), (queryImageBGR.shape[1] - 350, queryImageBGR.shape[0] - 150), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 3)
        cv2.putText(queryImageBGR, "%.1fft" % (dist / 12),(queryImageBGR.shape[1] - 350, queryImageBGR.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
        cv2.polylines(queryImageBGR, [np.int32(queryBorder)],True,(0,255,0),5, cv2.LINE_AA)

    else:
        print('Not enough matches')


    #   scanning for the second training image
    matches2 = flann.knnMatch(queryDescriptors, tDesc2, 2)
    for m,n in matches2:
        if(m.distance < 0.75*n.distance):
            goodMatch2.append(m)
    if(len(goodMatch2)>MIN_MATCH_COUNT):
        tp2 = []
        qp2 = []
        for m in goodMatch2:
            tp2.append(tKP2[m.trainIdx].pt)
            qp2.append(queryKP[m.queryIdx].pt)
        tp2,qp2 = np.float32((tp2,qp2))
        H, status = cv2.findHomography(tp2, qp2, cv2.RANSAC, 3.0)
        h, w = trainImage2.shape
        trainBorder = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        marker = (queryBorder[3][0][0] - queryBorder[0][0][0])

		#Finding the coordinates of the object in inches
        x0 = (queryBorder[1][0][0] - 640)/128
        y0 = (360 - queryBorder[1][0][1])/128
        x1 = (queryBorder[2][0][0] - 640)/128
        y1 = (360 - queryBorder[2][0][1])/128
        x2 = (queryBorder[0][0][0] - 640)/128
        y2 = (360 - queryBorder[0][0][1])/128
        x3 = (queryBorder[3][0][0] - 640)/128
        y3 = (360 - queryBorder[3][0][1])/128

        a = ((x1 - x0)*(y3 - y1) - (x3 - x1)*(y1 - y0))/((x3 - x1)*(y2 - y1) - (x2 - x1)*(y3 - y1))
        b = ((x1 - x0)*(y2 - y1) - (x2 - x1)*(y1 - y0))/((x3 - x1)*(y2 - y1) - (x2 - x1)*(y3 - y1))

        t0 = KNOWN_HEIGHT/math.sqrt((a*x2 - x0)*(a*x2 - x0) + (a*y2 - y0)*(a*y2 - y0) + focalLength*focalLength*(a - 1)*(a - 1)/(128*128))
        t2 = a*t0
        t3 = b*t0
        t1 = t0 + t3 - t2

        x0 = t0*x0
        x1 = t1*x1
        x2 = t2*x2
        x3 = t3*x3
        y0 = t0*y0
        y1 = t1*y1
        y2 = t2*y2
        y3 = t3*y3
        z0 = t0*focalLength/128
        z1 = t1*focalLength/128
        z2 = t2*focalLength/128
        z3 = t3*focalLength/128

        cx2 = (x1 + x2 + x3 + x0)/4
        cy2 = (y1 + y2 + y3 + y0)/4
        cz2 = (z0 + z1 + z2 + z3)/4

        dist = math.sqrt(cx2*cx2 + cy2*cy2 + cz2*cz2)

        cv2.putText(queryImageBGR, "(%.2f, %.2f, %.2f)" % (cx2, cy2, cz2), (queryImageBGR.shape[1] - 900, queryImageBGR.shape[0] - 150), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 3)
        cv2.putText(queryImageBGR, "%.1fft" % (dist / 12),(queryImageBGR.shape[1] - 900, queryImageBGR.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 0, 0), 3)
        cv2.polylines(queryImageBGR, [np.int32(queryBorder)],True,(255,0,0),5, cv2.LINE_AA)
    else:
        print('Not enough matches - 2')

	#plotting the points
    x1 = [0, cx1]
    y1 = [0, cy1]
    z1 = [0, cz1]

    x2 = [0, cy1]
    y2 = [0, cy2]
    z2 = [0, cz2]

    cx = [0]
    cy = [0]
    cz = [0]

    ax.set_xlim([-50,50])
    ax.set_ylim([-50,50])
    ax.set_zlim([0,300])

    ax.plot(x1, y1, z1, color='g', marker='o')
    ax.plot(x2, y2, z2, color='b', marker='o')
    ax.plot(cx, cy, cz, color = 'black', marker = 'o')
    plt.draw()
    plt.pause(0.001)
    ax.cla()
    queryImageBGR = cv2.resize(queryImageBGR, (1280, 720))
    cv2.imshow('result', queryImageBGR)
    if cv2.waitKey(1) == ord('q'):
         break

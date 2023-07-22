from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import utlis
import matplotlib.pyplot as plt
from imutils import paths

webcam = True
path = '1.jpg'
cap = cv2.VideoCapture(1)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale = 3
wP = 210 *scale
hP= 297 *scale

image = cv2.imread('ref.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)


#Laplacian Deriviative
edged = cv2.Laplacian(gray, -1, ksize=5, scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cv2.imshow('asdsa',edged)

cnts  = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
	# Get rect
	rect = cv2.minAreaRect(c)
	(x, y), (w, h), angle = rect
	# Display rectangle
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
	cv2.polylines(image, [box], True, (255, 0, 0), 2)

cv2.imshow('image',image)
cv2.waitKey(0)
	
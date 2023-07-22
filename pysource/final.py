from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
from imutils import paths
from object_detector import *

webcam = True
path = '1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def detect_objects(frame):
		# Convert Image to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Create a Mask with adaptive threshold
		mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

		edged = cv2.dilate(mask, None, iterations=3)
		edged = cv2.erode(edged, None, iterations=3)
	
		# Find contours
		contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		#cv2.imshow("mask", mask)
		objects_contours = []

		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > 2000:
				#cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
				objects_contours.append(cnt)

		return objects_contours,mask,edged

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)



while True:
	_, img = cap.read()

	corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
	contours,mask,edged = detect_objects(img)
	if corners:

		# Draw polygon around the marker
		int_corners = np.int0(corners)
		cv2.polylines(img, int_corners, True, (232, 192, 73), 10)

		# Aruco Perimeter
		aruco_perimeter = cv2.arcLength(corners[0], True)

		# Pixel to cm ratio
		pixel_cm_ratio = aruco_perimeter / 20

		for cnt in contours:
			# Get rect
			rect = cv2.minAreaRect(cnt)
			(x, y), (w, h), angle = rect

			# Get Width and Height of the Objects by applying the Ratio pixel to cm
			object_width = w / pixel_cm_ratio
			object_height = h / pixel_cm_ratio
			
			# Display rectangle
			box = cv2.boxPoints(rect)
			box = np.int0(box)

			cv2.circle(img, (int(x), int(y)), 7, (0, 0, 255), -1)
			cv2.polylines(img, [box], True, (255, 0, 0), 2)
			cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (115, 252, 173), 2)
			cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 20)), cv2.FONT_HERSHEY_PLAIN, 2, (115, 252, 173), 2)


	img = cv2.resize(img,(0,0),None,0.65,0.65)
	mask = cv2.resize(mask,(0,0),None,0.65,0.65)
	edged = cv2.resize(edged,(0,0),None,0.65,0.65)
	cv2.imshow('edged',edged)
	cv2.imshow('mask',mask)
	cv2.imshow('Image',img)

	if cv2.waitKey(1)& 0xFF == ord('q'):
		break 
	
cap.release()
cv2.destroyAllWindows()
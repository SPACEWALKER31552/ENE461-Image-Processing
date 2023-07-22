from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import utlis
import matplotlib.pyplot as plt

###################################

webcam = True
path = '1.jpg'
cap = cv2.VideoCapture(1)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale = 3
wP = 210 *scale
hP= 297 *scale

###################################

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


pixelsPerMetric = (120/2) * (16/30)

# ppm = ppm * (dispresemt*pixelnex)/disnew
################################### 
def calibrate() :

	if webcam:success,img = cap.read()
	else: img = cv2.imread(path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
	edged = cv2.dilate(edged, None, iterations=2)
	edged = cv2.erode(edged, None, iterations=2)
	cnts  = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	try :
		(cnts, _) = contours.sort_contours(cnts)
	except:
		print("Error at Calibrate Pos 1")
	finally:
		pass
	for c in cnts:
		if cv2.contourArea(c) < 100:
			continue
		c = max(cnts, key = cv2.contourArea)
		# print(c)
		orig = img.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)

		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)

		# draw the midpoints on the image
		cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)    
	


		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


		pixelsPerMetric = None
		if pixelsPerMetric is None:
			pixelsPerMetric = dB / 2.0


		return pixelsPerMetric

pixelsPerMetric = calibrate()
i=1
while	True:

	if webcam:success,img = cap.read()
	else: img = cv2.imread(path)


	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (7, 7), 0)


	# edged = cv2.Canny(gray, 50, 100)
	# edged = cv2.dilate(edged, None, iterations=1)
	# edged = cv2.erode(edged, None, iterations=1)
	# edged = cv2.dilate(edged, None, iterations=2)
	# edged = cv2.erode(edged, None, iterations=2)


	# cnts  = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)

	# cnts = max(cnts, key = cv2.contourArea)
	# cnts = cv2.minAreaRect(cnts)

	# kernel = np.ones((21, 21), np.uint8)
	# closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	# edges = cv2.Canny(closing, 50, 120)
	# edges = cv2.resize(edges,(0,0),None,0.5,0.5)
	# cv2.imshow(new,edges)

	try :
		(cnts, _) = contours.sort_contours(objects_contours)
	except:
		print("Error at Main Loop Pos 1")
	finally:
		pass
	# pixelsPerMetric = None

	for c in objects_contours:

		if cv2.contourArea(c) < 100:
			continue


		# c = max(objects_contours, key = cv2.contourArea)
		orig = img.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")


		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	
		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
	
		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
	
		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
	
		# draw the midpoints on the image
		cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	

		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)    
	   
	
  
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	
		
		# if pixelsPerMetric is None:
		#       pixelsPerMetric = dB / (args["width"])   



		pixelsPerMetric2 = None
		if pixelsPerMetric2 is None:
			pixelsPerMetric2 = dB / 2.0

		# compute the size of the object
		dimA = (dA / pixelsPerMetric)
		dimB = (dB / pixelsPerMetric)
	
		# draw the object sizes on the image
		cv2.putText(orig, "{:.2f}cm".format(dimA),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)
		cv2.putText(orig, "{:.2f}cm".format(dimB),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)

	
		# show the output image
		# cv2.imshow("Image", orig)
		
		i=i+1	
	try :
		orig = cv2.resize(orig,(0,0),None,0.5,0.5)
		img = cv2.resize(img,(0,0),None,0.5,0.5)
		edged = cv2.resize(edged,(0,0),None,0.5,0.5)
	except:
		print('Error at Main Loop Pos 2')
	finally:
		pass
	
	cv2.imshow('Detected PIC',orig)
	cv2.imshow('Webcam',img)
	cv2.imshow('Edge',edged)

	print("2 = ",pixelsPerMetric2)
	print("1 = ",pixelsPerMetric)
	# cv2.imwrite("example"+str(i)+".png",orig)
	
	if cv2.waitKey(1)& 0xFF == ord('q'):
		break 
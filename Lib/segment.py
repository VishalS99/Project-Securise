import argparse
import cv2
import numpy as np 
import functools

ap= argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="Path to the image")
args=vars(ap.parse_args())

image =cv2.imread(args["image"])
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fil=cv2.GaussianBlur(gray, (5,5), 0)
thresh=cv2.adaptiveThreshold(fil, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)

#remove unwanted whites
_, labels= cv2.connectedComponents(thresh)
mask= np.zeros(thresh.shape, dtype="uint8")

total_pixels = image.shape[0] * image.shape[1]
lower = total_pixels // 90 
upper = total_pixels // 20

for (i, label) in enumerate(np.unique(labels)):
  
    if label == 0:
        continue
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
 
    # If the number of pixels in the component is between lower bound and upper bound, 
    # add it to our mask
    if numPixels > lower and numPixels < upper:
        mask = cv2.add(mask, labelMask)

#contours 

cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c) for c in cnts]

def compare(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > 10:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]
boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

for rect in boundingBoxes:
    x,y,w,h = rect
    cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)

cv2.imshow('Final', image)
cv2.waitKey(0)


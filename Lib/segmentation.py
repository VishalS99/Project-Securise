# ML libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# DL libraries

import tensorflow as tf
from tensorflow import keras
import pytesseract

# Image processing

import glob
import cv2
import matplotlib
from PIL import Image
from imutils import contours
import os
import re


def character_segmentation(image,j):
    # separate coordinates from box
    
    
    #xmin, ymin, xmax, ymax = coords
    
    
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    
    
    #box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    
    
    # grayscale region within bounding box
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu Threshold", thresh)
    #cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    #cv2.imshow("Dilation", dilation)
    #cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""

    ROI_number=0
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.5: continue

        # if width is not wide enough relative to total width then skip
        #if width / float(w) > 15: continue

        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        roi = cv2.medianBlur(roi, 5)
        roi = cv2.resize(roi,(30,30))
        try:
         text = pytesseract.image_to_string(roi,config='-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz  --psm 10')
        # clean tesseract text by removing any unwanted blank spaces
         clean_text = re.sub('[\W_]+', '', text)
        
         plate_num += clean_text[0]
         print(clean_text[0])
        except: 
         text = None
        cv2.imwrite('extracted_images/image_{}_ROI_{}.png'.format(j, ROI_number), roi)
        ROI_number += 1      
    if plate_num != None:
        print("License Plate #: ", plate_num)
    #cv2.imshow("Character's Segmented", im2)
    #cv2.waitKey(0)
    return plate_num


#define path

for i in range(1,2):
 path='test{}.png'.format(i)
 image= cv2.imread(path)


#call character_segmentation function
 character_segmentation(image,i)


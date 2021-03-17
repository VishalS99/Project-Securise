# ML libraries

from CharacterDetection.detect_model import Net
from torchvision import transforms
import numpy as np
from skimage.segmentation import clear_border
# DL libraries

import pytesseract
from PIL import Image, ImageFilter
# Image processing


import torch

import cv2
import os
import re


def angle_with_start(coord, start):
    vec = coord - start
    return np.angle(np.complex(vec[0][0], vec[0][1]))


def sort_clockwise(points):
    # function to sort the 2D points in clockwise
    coords = sorted(points, key=lambda coord: np.linalg.norm(coord))
    start = coords[0]
    rest = coords[1:]
    rest = sorted(rest, key=lambda coord: angle_with_start(
        coord, start[0]), reverse=True)
    rest.insert(0, start)
    return rest


def border(image, original, ratio, j):
    # function which returns the cropped image of the number plate

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    #image = cv2.erode(image,rect_kern,iterations = 2)
    image_blur = cv2.bilateralFilter(image, 11, 90, 90)
    w = int(250*ratio)
    h = int(250)
    image_area = w*h
    edges = cv2.Canny(image_blur, 60, 120)

    ext_path = "Images/segmentation{}".format(j)
    cv2.imwrite(ext_path+'/image_edge.png', edges)
    cnts, hierarchy = cv2.findContours(
        edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    im2 = image.copy()
    _ = cv2.drawContours(im2, cnts, -1, (255, 0, 255), 2)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    plate = None

    mask = np.zeros_like(original)

    om = original.copy()
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        ec = cv2.approxPolyDP(c, 0.02*perimeter, True)

        if len(ec) == 4:
            area = cv2.contourArea(c)
            if area/image_area < 0.3:
                continue

            pts = sort_clockwise(ec)
            pts1 = np.float32([pts[0][0], pts[3][0], pts[1][0], pts[2][0]])
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            _ = cv2.drawContours(om, c, -1, (255, 0, 255), 2)
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            final_image = cv2.warpPerspective(image, matrix, (int(w), h))
            final_image_orig = cv2.warpPerspective(
                original, matrix, (int(w), h))

            # cv2.imwrite(ext_path+'/image_borders.png', om)
            cv2.imwrite(ext_path+'/image_crop.png', final_image)
            cv2.imwrite(ext_path+'/image_crop_orig.png', final_image_orig)
            return final_image, final_image_orig, 1
    return image, original, 0


def detection(sorted_contours, image, bound, j, original):
    # character identification
    # bound: (r1_l ,r2_h ,ratio_h,ratio_l,area_bound)
    ext_path = os.getcwd() + "/Output/segmentation{}".format(j)
    if os.path.isdir(ext_path) == False:
        os.mkdir(ext_path)
    # _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    # blur = cv2.GaussianBlur(image, (5, 5), 0)
    # ret3, image = cv2.threshold(
    #     blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imwrite(ext_path + '/image_perspective_transformed.png', image)
    im = image.copy()
    # original_copy = original.copy()
    plate_num = ""
    ROI_n = 0
    roi_total = []
    # print("image", image)
    # print("CNT: ", len(sorted_contours))
    character_model = torch.load(
        "Lib/CharacterDetection/model_character_detect.pt")
    label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                  'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'fail']
    transform = transforms.Compose(
        [
            #  transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, _ = im.shape[:2]

        r1 = height / float(h)
        ratio = h/float(w)
        area = w*h
        flag = 0
        if r1 < bound[0] or r1 > bound[1]:
            continue

        if ratio < bound[2] or ratio > bound[3]:
            continue

        if area < bound[4]:
            continue

        # if not roi_total:
        #     roi_total.append(x)
        #     roi_total.append(y)
        #     roi_total.append(w)
        #     roi_total.append(h)
        # elif (x > roi_total[0] and y >= roi_total[1] and (w) <= roi_total[2] and h <= roi_total[3]) or ((w*h) <= roi_total[2]*roi_total[3]):
        #     flag = 1
        # else:
        #     roi_total[0] = x
        #     roi_total[1] = y
        #     roi_total[2] = w
        #     roi_total[3] = h

        # if flag:
        #     continue
        rect = cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 5)
        cv2.imwrite(ext_path+'/image_cnt.png', rect)
        roi = image[y-5:y+h+5, x-5:x+w+5]

        white = [255, 255, 255]
        # roi = cv2.copyMakeBorder(
        #     roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=white)

        # roi = cv2.resize(roi, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        KK = np.ones((3, 3), np.uint8)
        roi = cv2.dilate(roi, KK, iterations=1)
        # roi = cv2.threshold(cv2.bilateralFilter(roi, 5, 75, 75), 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        roip = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        roip = Image.fromarray(roip)
        roip = roip.resize((28, 28))

        try:
            roip = transform(roip)
            roip = roip.cuda()
            lp = character_model(roip[None, ...])
            ps = torch.exp(lp)
            probab = list(ps.cpu()[0])
            pred_label = probab.index(max(probab))
            text = label_list[pred_label]
            if text == 'fail':
                continue
            if len(plate_num) <= 4 and len(plate_num) >= 2 and text == 'O' or text == 'o':
                text = '0'
            if len(plate_num) >= 6 and text == 'O' or text == 'o':
                text = '0'
            if len(plate_num) <= 4 and len(plate_num) >= 2 and text == 'S' or text == 's':
                text = '5'
            if len(plate_num) >= 6 and text == 's' or text == 'S':
                text = '5'
            plate_num += text[0].capitalize()
        except:
            print("Err instantiating model\n")
            print("Exiting... \n")
            continue

        cv2.imwrite(ext_path+'/image_{}_ROI_{}.png'.format(j, ROI_n), roi)
        ROI_n += 1
        cv2.imwrite(ext_path + '/image_final.png', rect)
    if plate_num and plate_num[0] == 'T':
        s = list(plate_num)
        s[1] = 'N'
        plate_num = "".join(s)
    return plate_num


def character_segmentation(image, j):
    # character segmentation and detection

    img_size = image.shape
    ext_path = os.getcwd() + "/Output/segmentation{}".format(j)
    if os.path.isdir(ext_path) == False:
        os.mkdir(ext_path)

    ratio = img_size[1]/img_size[0]
    image = cv2.resize(image, (int(250*ratio), int(250)))

    # image = image * 2
    cv2.imwrite(ext_path + '/image_initial.png', image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    cv2.imwrite(ext_path + '/image_gray.png', gray)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    thresh = cv2.threshold(
        gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    res = opening
    res = cv2.erode(res, rect_kern, iterations=1)
    cv2.imwrite(ext_path + '/image_thresh.png', res)

    image_crop, image_crop_orig, crop_flag = border(res, gray, ratio, j)
    dilation = cv2.dilate(image_crop, rect_kern, iterations=2)
    cv2.imwrite(ext_path + '/image_dilation1.png', dilation)
    white = [255, 255, 255]
    dilation = cv2.copyMakeBorder(
        dilation, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white)
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(
        contours, key=lambda ctr: cv2.minAreaRect(ctr)[0])

    if crop_flag == 1:
        bound = [0, 6, 1.3, 1000, 2500]
    else:
        bound = [0, 6, 1.3, 4, 1800]
    return detection(sorted_contours, dilation, bound, j, image_crop_orig)

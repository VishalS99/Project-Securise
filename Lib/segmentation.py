# ML libraries

import numpy as np

# DL libraries

import pytesseract

# Image processing

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

            cv2.imwrite(ext_path+'/image_borders.png', om)
            cv2.imwrite(ext_path+'/image_crop.png', final_image)
            cv2.imwrite(ext_path+'/image_crop_orig.png', final_image_orig)
            return final_image, final_image_orig


def detection(sorted_contours, image, bound, j):
    # character identification
    # bound: (r1_l ,r2_h ,ratio_h,ratio_l,area_bound)
    ext_path = os.getcwd() + "/Images/segmentation{}".format(j)
    if os.path.isdir(ext_path) == False:
        os.mkdir(ext_path)
    # _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret3, image = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=5)
    cv2.imshow("Processed perspective transformed", image)
    cv2.waitKey(0)
    im = image.copy()
    plate_num = ""
    ROI_n = 0
    roi_total = []
    # print("image", image)
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = im.shape[:2]
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

        if not roi_total:
            roi_total.append(x)
            roi_total.append(y)
            roi_total.append(w)
            roi_total.append(h)
        elif (x > roi_total[0] and y >= roi_total[1] and (w) <= roi_total[2] and h <= roi_total[3]) or ((w*h) <= roi_total[2]*roi_total[3]):
            flag = 1
        else:
            roi_total[0] = x
            roi_total[1] = y
            roi_total[2] = w
            roi_total[3] = h

        if flag:
            continue
        roi = image[y:y+h, x:x+w]
        #roi = cv2.bitwise_not(roi)
        #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        white = [255, 255, 255]
        roi = cv2.copyMakeBorder(
            roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=white)
        kernel = np.ones((5, 5), np.uint8)

        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        rect = cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 5)

        roi = cv2.resize(roi, (45, 65))

        try:
            text = pytesseract.image_to_string(
                roi, config='-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz  --psm 10')

        # clean tesseract text by removing any unwanted blank spaces
            text = re.sub('[\W_]+', '', text)
            if(text[0] == 'g'):
                plate_num += '9'
            else:
                plate_num += text[0]
        except:
            text = None
            if(ROI_n == 1 and plate_num[0] == 'T'):
                text = 'N'
                plate_num += text[0]

        # print("Plate", text)
        cv2.imwrite(ext_path+'/image_{}_ROI_{}.png'.format(j, ROI_n), roi)
        ROI_n += 1
        cv2.imwrite(ext_path + '/image_final.png', rect)
    if plate_num[0] == 'T':
        s = list(plate_num)
        s[1] = 'N'
        plate_num = "".join(s)
    return plate_num


def character_segmentation(image, j):
    # character segmentation and detection

    img_size = image.shape
    ext_path = os.getcwd() + "/Images/segmentation{}".format(j)
    if os.path.isdir(ext_path) == False:
        os.mkdir(ext_path)

    ratio = img_size[1]/img_size[0]
    image = cv2.resize(image, (int(250*ratio), int(250)))

    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    cv2.imwrite(ext_path + '/image_initial.png', image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(ext_path + '/image_gray.png', gray)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(ext_path + '/image_thresh.png', thresh)

    crop_flag = 1
    # kernel = np.ones((5,5),np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    dilation = cv2.erode(thresh, rect_kern, iterations=1)
    image_crop, image_crop_orig = border(dilation, gray, ratio, j)
    if image_crop is None:
        image_crop = dilation
        crop_flag = 0
    cv2.imwrite(ext_path + '/image_dilation1.png', dilation)
    try:
        contours, hierarchy = cv2.findContours(
            image_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(
            image_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(
        contours, key=lambda ctr: cv2.minAreaRect(ctr)[0])

    if crop_flag == 1:
        bound = [0, 6, 1.3, 1000, 2500]
    else:
        bound = [0, 6, 1.3, 4, 1800]

    # print(bound)
    return detection(sorted_contours, image_crop_orig, bound, j)


# for i in range(4, 10):
#     path = 'test{}.png'.format(i)
#     image = cv2.imread(path)
#     print('Plate number# --', character_segmentation(image, i))

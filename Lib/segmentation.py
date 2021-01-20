import pytesseract
import cv2
import re
import os


def character_segmentation(image, j):

    img_size = image.shape
    # resize image to (250, 250) relative to original original for better readability
    ratio = img_size[1]/img_size[0]
    image = cv2.resize(image, (int(250*ratio), int(250)))

    # grayscale region within bounding box
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)

    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(
            dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(
            dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours left-to-right
    sorted_contours = sorted(
        contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # create copy of gray image
    im2 = gray.copy()

    # create blank string to hold license plate number
    plate_num = ""

    ROI_number = 0

    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = im2.shape

        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6:
            continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1:
            continue

        # if width is not wide enough relative to total width then skip
        # if width / float(w) > 15: continue

        area = w*h
        print("Character{} :".format(ROI_number), area)
        if area < 2000:
            continue

        rect = cv2.rectangle(im2, (x, y), (x+w, y+h), (0, 255, 0), 5)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        roi = cv2.medianBlur(roi, 5)
        roi = cv2.resize(roi, (30, 30))
        cv2.imshow("{}".format(ROI_number), rect)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

        try:
            text = pytesseract.image_to_string(
                roi, config='-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz  --psm 10')

        # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)

            plate_num += clean_text[0]
        except:
            text = None
            print("Plate", text)
        ext_path = "./Lib/Extracted_NP_Characters"
        if os.path.isdir(ext_path) == False:
            os.mkdir(ext_path)
        cv2.imwrite(ext_path + '/image_final.png', rect)
        cv2.imwrite(
            ext_path + '/image_{}_ROI_{}.png'.format(j, ROI_number), roi)
        ROI_number += 1

    if plate_num != None:
        print("License Plate #: ", plate_num)
    return plate_num

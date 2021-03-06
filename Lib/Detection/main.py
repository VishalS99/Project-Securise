import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import io
from datetime import datetime
import os


def yolo_detect(img, height, width, channels):
    # yolo
    net = cv2.dnn.readNet(os.path.join(
        os.getcwd(), "Lib/Detection/yolov3.weights"), os.path.join(os.getcwd(), "Lib/Detection/yolov3.cfg"))
    classes = []

    with open(os.path.join(
            os.getcwd(), "Lib/Detection/coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    colors = (0, 255, 0)

    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    max_area = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.75:

                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                if max_area < h*w:
                    max_area = h*w
                    # print(max_area)
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Getting the biggest area Objects Alone and Discarding the others
    max_area_boxes = []
    new_confidence = []
    for i in range(len(boxes)):
        if boxes[i][2]*boxes[i][3] == max_area:
            max_area_boxes.append(boxes[i])
            new_confidence.append(confidences[i])

    # Marking the bounding Boxes for the largest Area Objects
    indexes = cv2.dnn.NMSBoxes(max_area_boxes, new_confidence, 0.5, 0.4)

    return max_area_boxes, indexes, classes, class_ids


def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v


def visualize_detection(img, boxes, indexes, classes, class_ids):
    cropped_image = 0
    label = 0
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Uncomment the below line to show label on image
            cv2.putText(img, label, (x, y-5), font, 1, color, 2)

            # Crop out unwanted region
            cropped_image = img[y+5:y+h-5, x+5:x+w-10]
    
    cropped_image = cv2.resize(cropped_image, None, fx=1, fy=1)
    return cropped_image, label


def yolo_detection(img):
    # Loading image
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    max_area_boxes, indexes, classes, class_ids = yolo_detect(
        img, height, width, channels)

    cropped_image, label = visualize_detection(img,
                                               max_area_boxes, indexes, classes, class_ids)

    cv2.imwrite("../../Dataset/Cropped_Vehicle.jpg", cropped_image)

    # Getting the dominant Color
    data = np.reshape(cropped_image, (-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        data, 1, None, criteria, 10, flags)

    # Converting to HSV color space from RGB.
    r = (centers[0].astype(np.int32))[2]
    g = (centers[0].astype(np.int32))[1]
    b = (centers[0].astype(np.int32))[0]
    h, s, v = rgb_to_hsv(r, g, b)

    # Printing All the Details Analysed

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return h, v, s, dt_string, label, cropped_image



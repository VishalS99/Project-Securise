import cv2
import numpy as np

# Loading Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("auto.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
max_area=0

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            if max_area<h*w:
                max_area=h*w
                print(max_area)
            confidences.append(float(confidence))
            class_ids.append(class_id)

#Getting the biggest area Objects Alone and Discarding the others
max_area_boxes=[]
new_confidence=[]
for i in range(len(boxes)):
    if boxes[i][2]*boxes[i][3]==max_area:
        max_area_boxes.append(boxes[i])
        new_confidence.append(confidences[i])

#Marking the bounding Boxes for the largest Area Objects
indexes = cv2.dnn.NMSBoxes(max_area_boxes, new_confidence, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(max_area_boxes)):
    if i in indexes:
        x, y, w, h = max_area_boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

cv2.imshow("Image", img)

#Extracting the details from the bounding box
#Color yet to do
#Name yet to do

cv2.waitKey(0)
cv2.destroyAllWindows()
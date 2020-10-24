from os import close
import json
import re
from PIL import Image

DIR = "/home/vishals/Dataset/UFPR-ALPR/UFPR-ALPR dataset/training"
FINAL = "data_coco_train.json"


def write_json(data, filename=FINAL):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# ===================================================
# ================ Images Structure =================
# ===================================================
#
#  {
#             "id": 0,
#             "license": 1,
#             "file_name": "0001.jpg",
#             "height": 275,
#             "width": 490,
#             "date_captured": "2020-07-20T19:39:26+00:00"
#         },


def add_image_data(filename=FINAL):
    ID = 0
    with open(filename) as json_file:
        data = json.load(json_file)
        temp = data['images']
        for i in range(60):
            INTER_DIR = DIR + "/track00" + \
                (("0" + str(i+1)) if i < 9 else str(i+1))

            for j in range(30):
                FILE_ABS = INTER_DIR + "/track00" + \
                    (("0" + str(i+1)) if i < 9 else str(i+1)) + \
                    "[" + ("0" + str(j + 1) if j < 9 else str(j+1)) + "]"
                FILE_LCL = "../Dataset/data_files/track00" + \
                    (("0" + str(i+1)) if i < 9 else str(i+1)) + \
                    "[" + ("0" + str(j + 1) if j < 9 else str(j+1)) + "].png"

                LINE = FILE_ABS + ".png"
                im = Image.open(LINE)
                (width, height) = im.size
                xx = {
                    "id": ID,
                    "license": 1,
                    "file_name": FILE_LCL,
                    "height": height,
                    "width": width,
                    "date_captured": "2020-07-20T19:39:26+00:00"
                }
                ID = ID + 1

                print(ID)
                temp.append(xx)

        write_json(data)

# ===================================================
# ============== Annotations Structure ==============
# ===================================================


#         "annotations": [
#     {
#       "id": 1,
#       "bbox": [
#         100,
#         116,
#         140,
#         170
#       ],
#       "image_id": 0,
#       "segmentation": [],
#       "ignore": 0,
#       "area": 23800,
#       "iscrowd": 0,
#       "category_id": 0
#     }
#   ]

def add_annotations_data(filename=FINAL):
    ID = 0
    with open(filename) as json_file:
        j_data = json.load(json_file)
        temp = j_data['annotations']
        for i in range(60):
            INTER_DIR = DIR + "/track00" + \
                (("0" + str(i+1)) if i < 9 else str(i+1))

            for j in range(30):
                FILE_ABS = INTER_DIR + "/track00" + \
                    (("0" + str(i+1)) if i < 9 else str(i+1)) + \
                    "[" + ("0" + str(j + 1) if j < 9 else str(j+1)) + "]"

                LINE = "" + FILE_ABS + ".png"
                f = open(FILE_ABS + ".txt", "rt")
                data = f.readlines()[7][16:].replace(" ", ",").split("\n")
                ld = []
                for l in data:
                    if l != "":
                        ld = l.split(",")
                        ld[0] = int(ld[0])
                        ld[1] = int(ld[1])
                        ld[2] = int(ld[2])
                        ld[3] = int(ld[3])
                im = Image.open(LINE)
                (width, height) = im.size
                im_size = width * height
                xx = {
                    "id": ID,
                    "image_id": ID,
                    "bbox": ld,
                    "segmentation": [],
                    "area": im_size,
                    "iscrowd": 0,
                    "category_id": 1
                }
                ID = ID + 1

                print(xx)
                temp.append(xx)

        write_json(j_data)


if __name__ == '__main__':
    add_image_data()
    add_annotations_data()

import cv2
import os
from sys import path
import  random
from detectron2.data.catalog import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
os.chdir(os.path.join(os.getcwd(), "Lib/"))
path.append("../")
from Dataset.get_dataset import load_dataset

os.chdir("..")


(train_dataset_dicts, NP_train_metadata, _, _) = load_dataset(os.path.join(os.getcwd(),"Dataset/data_coco_train.json"), os.path.join(os.getcwd(),"Dataset/data_coco_test.json"))
# for d in random.sample(train_dataset_dicts, 1):
#     img = cv2.imread(d["file_name"])
#     print(img)
#     visualizer = Visualizer(
#         img[:, :, ::-1], metadata=NP_train_metadata, scale=0.4, instance_mode=ColorMode.SEGMENTATION)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("Image", out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)



cfg = get_cfg()   
cfg.merge_from_file(os.path.join(os.getcwd(),"Lib/Config/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.WEIGHTS = os.path.join("Results", "model_final.pth")
predictor = DefaultPredictor(cfg)

im = cv2.imread('/home/vishals/projects/Securise/Dataset/data_files/track0001[02].png')
outputs = predictor(im)
print("#### Output {}".format(outputs))
# v = Visualizer(im[:, :, ::-1],
#             metadata=NP_train_metadata, 
#             scale=0.8
#                 )
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow("image",out.get_image()[:, :, ::-1])
# cv2.waitKey(0)
from detectron2.data import transforms as T
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine.defaults import DefaultPredictor
from Config.model_config import load_config
from Detection.main import yolo_detection
from segmentation import *
import cv2
import os
import numpy as np
from sys import path
path.append("../")
# setup_logger()


def test():
    # ---------------------------------------------------------------
    # Loading configuration of the model
    # ---------------------------------------------------------------
    print("- Loading Model configuration")
    (cfg, metadata) = load_config(True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)

    # ---------------------------------------------------------------
    # Load image to test
    # ---------------------------------------------------------------
    print("- Reading test image")
    try:
        img = cv2.imread(
            "/home/vso/Projects/Securise/Dataset/dataset1/54.jpg")
    except Exception as e:
        print("Err: {}".format(e))
        exit()

    # ---------------------------------------------------------------
    # Vehicle extraction - yolo
    # ---------------------------------------------------------------
    print("- Predicting properties of the vehicle")
    (h, s, v, dt_string, label, cimg) = yolo_detection(img)

    # ---------------------------------------------------------------
    # Extracting number plate
    # ---------------------------------------------------------------
    print("- Predicting Bounding boxes")
    output = predictor(cimg)
    inst = output["instances"]

    visualizer = Visualizer(
        cimg[:, :, ::-1], metadata=metadata, scale=0.7, instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_instance_predictions(inst.to("cpu"))
    cv2.imwrite(os.path.join(os.getcwd(), "Demo/res2.jpg"),
                out.get_image()[:, :, ::-1])
    # cv2.imshow("ROI", out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ---------------------------------------------------------------
    # Obtain ROI
    # ---------------------------------------------------------------
    print("- Generating ROI")
    boxes = inst.pred_boxes.tensor.detach().cpu().numpy()[0].astype(np.int32)
    roi = cimg[boxes[1]:boxes[3], boxes[0]:boxes[2]]
    roi = cv2.detailEnhance(roi, sigma_s=15, sigma_r=0.5)

    # ---------------------------------------------------------------
    # Perform Character Segmentation
    # ---------------------------------------------------------------
    print("- Performing character segmentation")
    np_character = character_segmentation(roi, 2)
    print("###############################")
    print("## Properties of the vehicle ##")
    print("###############################")
    print("## Vehicle type: {}\n## Vehicle color: ({}, {}, {})\n## Vehicle entry time: {}".
          format(label, h, s, v, dt_string))
    print("## Number plate: ", np_character)
    print("- Done")


if __name__ == "__main__":
    test()

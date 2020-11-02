from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine.defaults import DefaultPredictor
from Config.model_config import load_config
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
    (cfg, data, metadata) = load_config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)

    # ---------------------------------------------------------------
    # Load image to test and produce output
    # ---------------------------------------------------------------
    print("- Reading test image")
    try:
        img = cv2.imread(
            "/home/vishals/projects/Securise/Demo/demo.png")
    except Exception as e:
        print("Err: {}", format(e))
        exit()

    print("- Predicting Bounding boxes")
    output = predictor(img)
    inst = output["instances"]

    visualizer = Visualizer(
        img[:, :, ::-1], metadata=metadata, scale=0.7, instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_instance_predictions(inst.to("cpu"))
    cv2.imwrite(os.path.join(os.getcwd(), "Demo/res2.jpg"),
                out.get_image()[:, :, ::-1])

    # ---------------------------------------------------------------
    # Obtain ROI
    # ---------------------------------------------------------------
    print("- Generating ROI")
    boxes = inst.pred_boxes.tensor.detach().cpu().numpy()[0].astype(np.int32)
    roi = img[boxes[1]:boxes[3], boxes[0]:boxes[2]]
    roi = cv2.detailEnhance(roi, sigma_s=15, sigma_r=0.5)

    # ---------------------------------------------------------------
    # Perform Character Segmentation
    # ---------------------------------------------------------------
    print("- Performing character segmentation")
    character_segmentation(roi, 5)
    print("- Done")


if __name__ == "__main__":
    test()

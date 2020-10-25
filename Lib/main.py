import os
from sys import path

from detectron2.data.catalog import Metadata
path.append("../")
from Config.model_config import load_config 
import cv2
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
setup_logger()



def test():
    (cfg, data, metadata) = load_config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    
    predictor = DefaultPredictor(cfg)

    img = cv2.imread("/home/vishals/Desktop/d2.jpg")
    output = predictor(img)
    print("###OUPUT: {}".format(output))

    visualizer = Visualizer(
        img[:, :, ::-1], metadata=metadata, scale=0.7, instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_instance_predictions(output["instances"].to("cpu"))
    cv2.imshow("Image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

if __name__ == "__main__":
    test()

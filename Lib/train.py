import os
from sys import path
path.append("../")
from Config.model_config import load_config 
import cv2
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
setup_logger()



def train():

    (cfg,_,_) = load_config()
    cfg.MODEL.WEIGHTS = os.path.join(os.getcwd(), "Lib/Model/model_final_f6e8b1.pkl")
    cfg.DATASETS.TRAIN = ("NP_Dataset_train",)
    cfg.DATASETS.TEST = ("NP_Dataset_test",)
    cfg.SOLVER.IMS_PER_BATCH = 2

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def validate():

    (cfg, _,bb) = load_config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.69
    
    predictor = DefaultPredictor(cfg)

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("NP_Dataset_test", cfg, False, output_dir="Output/")
    val_loader = build_detection_test_loader(cfg, "NP_Dataset_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    validate()
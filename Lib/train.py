import os
import random

from sys import path
import cv2

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
os.chdir(os.path.join(os.getcwd(), "Lib/"))
path.append("../")
from Dataset.get_dataset import load_dataset

os.chdir("..")
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
setup_logger()


def load_config():
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(os.getcwd(),"Lib/Config/faster_rcnn_R_101_FPN_3x.yaml"))
    (train_dataset_dicts, NP_train_metadata, _, _) = load_dataset(
        os.path.join(os.getcwd(),"Dataset/data_coco_train.json"), os.path.join(os.getcwd(),"Dataset/data_coco_test.json"))

    cfg.DATALOADER.NUM_WORKERS = 2
    
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = (300)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = "Results"
    cfg.CUDNN_BENCHMARK = True
    return (cfg,train_dataset_dicts, NP_train_metadata)

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

    (cfg, aa,bb) = load_config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    
    predictor = DefaultPredictor(cfg)

    # for d in random.sample(aa, 1):
    #     img = cv2.imread(d["file_name"])
    #     outp = predictor(img)
    #     visualizer = Visualizer(
    #         img[:, :, ::-1], metadata=bb, scale=0.4, instance_mode=ColorMode.SEGMENTATION)
    #     out = visualizer.draw_instance_predictions(outp["instances"].to("cpu"))
    #     cv2.imshow("Image", out.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
    img = cv2.imread("/home/vishals/projects/Securise/Dataset/data_files/track0048[14].png")
    outp = predictor(img)
    print("###OUPUT: {}".format(outp))
    visualizer = Visualizer(
        img[:, :, ::-1], metadata=bb, scale=0.7, instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_instance_predictions(outp["instances"].to("cpu"))
    cv2.imshow("Image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    # outputs = predictor(im)
    # print("#### Output {}".format(outputs))
    # v = Visualizer(im[:, :, ::-1],
    #             metadata=NP_train_metadata, 
    #             scale=0.8
    #                 )
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow("image",out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)

    # from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    # from detectron2.data import build_detection_test_loader

    # evaluator = COCOEvaluator("NP_Dataset_test", cfg, False, output_dir="Output/")
    # val_loader = build_detection_test_loader(cfg, "NP_Dataset_test")
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    train()
from detectron2.config import get_cfg
import os
from sys import path
os.chdir(os.path.join(os.getcwd(), "Lib/"))
path.append("../")
from Dataset.get_dataset import load_dataset
os.chdir("..")


def load_config(test=False):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(
        os.getcwd(), "Lib/Model/faster_rcnn_R_101_FPN_3x.yaml"))
    
    if test is True:
        (train_dataset_dicts, NP_train_metadata, _, _) = load_dataset(
            os.path.join(os.getcwd(), "Dataset/data_coco_train.json"), os.path.join(os.getcwd(), "Dataset/data_coco_test.json"), True)
    else:
        (train_dataset_dicts, NP_train_metadata, _, _) = load_dataset(
        os.path.join(os.getcwd(), "Dataset/data_coco_train.json"), os.path.join(os.getcwd(), "Dataset/data_coco_test.json"))
    
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = (300)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = "Results"
    cfg.CUDNN_BENCHMARK = True
    
    if test  is True:
        return (cfg, NP_train_metadata)

    return (cfg, train_dataset_dicts, NP_train_metadata)

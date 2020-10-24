from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import random


def load_dataset(train_path, test_path):

    # Training data
    register_coco_instances("NP_Dataset_train", {},
                            "data_coco_train.json", "data_files")

    NP_train_metadata = MetadataCatalog.get("NP_Dataset_train")
    NP_train_metadata.thing_colors = [(255, 0, 0), (0, 255, 0)]
    train_dataset_dicts = DatasetCatalog.get("NP_Dataset_train")
    print("## Loaded Training Data")
    # Training data
    register_coco_instances("NP_Dataset_test", {},
                            "data_coco_test.json", "data_files")

    NP_test_metadata = MetadataCatalog.get("NP_Dataset_test")
    NP_test_metadata.thing_colors = [(255, 0, 0), (0, 255, 0)]
    test_dataset_dicts = DatasetCatalog.get("NP_Dataset_test")
    print("## Loaded Testing Data")

    return (train_dataset_dicts, NP_train_metadata, test_dataset_dicts, NP_test_metadata)


def data_vis(train, test):
    (train_dataset_dicts, NP_train_metadata, _, _) = load_dataset(train, test)
    for d in random.sample(train_dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=NP_train_metadata, scale=0.4, instance_mode=ColorMode.SEGMENTATION)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("Image", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


if __name__ == "__main__":
    data_vis("data_coco_train.json", "data_coco_test.json")

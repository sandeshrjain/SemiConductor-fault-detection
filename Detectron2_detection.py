from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(), plt.imshow(im), plt.axis('off');
    
import os
import cv2
import pandas as pd
#MetadataCatalog.remove("fault_test.txt")
#MetadataCatalog.remove("fault_trainval.txt")
#DatasetCatalog.remove("fault_trainval.txt")
#DatasetCatalog.remove("fault_test.txt")
def get_fault_dicts(img_dir, d):
  train_file = os.path.join(img_dir, d)
  df = pd.read_csv(train_file, sep=" ", header=None)
  df[0] = df[0].str[:-4] + '_test.jpg'
  dataset_dicts = []
  for i in range(df[0].shape[0]):
    record = {}
    filename = os.path.join(img_dir, df[0][i])
    height, width = cv2.imread(filename).shape[:2]
    record["file_name"] = filename
    record["image_id"] = i
    record["height"] = height
    record["width"] = width

    ann_path = os.path.join(img_dir, df[1][i])
    annos = pd.read_csv(ann_path, sep=" ", header=None)

    objs = []

    for j in range(annos.shape[0]):
      obj = {
      "bbox": [annos[0][j], annos[1][j], annos[2][j], annos[3][j]],
      "bbox_mode": BoxMode.XYXY_ABS,
      #"segmentation": [poly],
      "category_id": annos[4][j]-1}
      objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
  return dataset_dicts

img_dir = "./Surface-Defect-Detection/DeepPCB/PCBData"
for d in ["trainval.txt", "test.txt"]:
    print(get_fault_dicts(img_dir, d))
    DatasetCatalog.register("fault_" + d, lambda d=d: get_fault_dicts(img_dir, d))
    MetadataCatalog.get("fault_" + d).set(thing_classes=["open", "short", "mousebite", "spur", "copper", "pin-hole"])

fault_metadata = MetadataCatalog.get("fault_trainval.txt")


import matplotlib.pyplot as plt
dataset_dicts = get_fault_dicts(img_dir, "trainval.txt")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fault_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])
    
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("fault_trainval.txt",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

predictor = DefaultPredictor(cfg)



from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_fault_dicts(img_dir, "test.txt")
for d in random.sample(dataset_dicts, 1):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=fault_metadata, 
                   scale=0.5 #, 
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])
    
    

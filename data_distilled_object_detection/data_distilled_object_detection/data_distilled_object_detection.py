#!/usr/bin/env python3
import pycocotools
# from data_distilled_object_detection.data_distilled_object_detection.data_distillation import multi_transform_inference, create_annotation_file, merge_illumination_annotations
from data_distillation import multi_transform_inference, create_annotation_file, merge_illumination_annotations, draw_boxes_on_image
from detectron2.utils.logger import setup_logger
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.data.datasets.coco import convert_to_coco_json
from xml.etree import ElementTree
import argparse
import cv2
import os
import warnings
from enum import Enum
from pathlib import Path
from customs import *
import torch
from from_root import from_root

warnings.simplefilter(action='ignore', category=FutureWarning)
# Detectron2 logger setup
setup_logger()
# Local logger setup 
path = Path(__file__)
logger = setup_logger(output=None, name=path.parent.name + "." + path.stem)

ANNOTATIONS_TYPE = "annotations_filtered"
TRAINABLE_SIZE = 160 #240

TRAIN_SET_PERCENTAGE = 0.7
VAL_SET_PERCENTAGE = 0.2
TEST_SET_PERCENTAGE = 0.1

# DISTILLED_DATA_SIZE = int(TRAINABLE_SIZE * 2.0) #20
DISTILLED_DATA_SIZE = int(TRAINABLE_SIZE)
# DISTILLED_DATA_SIZE = 1

TOTAL_DATA_DISTILLATION_SIZE = TRAINABLE_SIZE + DISTILLED_DATA_SIZE
NUM_VAL_EPOCH = 3

BASE_MODEL_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
EVAL_WEIGHTS_FILE = "model_final.pth"

TRAINING_MODE = "train"
EVALUATION_MODE = "evaluate"
PREDICTION_MODE = "predict"
BATCH_PREDICTION_MODE = "batch_predict"
INSPECTION_MODE = "inspect"
TUNING_MODE = "tune"
FPN_LAYERS_MODE = "viz_fpn_layers"
DATA_DISTILLATION_MODE = "data_distillation"
RETRAINING_MODE = "retrain"
ANNOTATION_MERGE_MODE = "annotation_merge"

DATASET_DIR_1 = str(from_root("datasets/ApplesAnnotated-800"))
DATASET_DIR_2 = str(from_root("datasets/ApplesAnnotated-1374"))


def extract_boxes(filename):
  # load and parse the file
  tree = ElementTree.parse(filename)
  # get the root of the document
  root = tree.getroot()
  # extract each bounding box
  boxes = list()
  for box in root.findall('.//bndbox'):
    xmin = int(box.find('xmin').text)
    ymin = int(box.find('ymin').text)
    xmax = int(box.find('xmax').text)
    ymax = int(box.find('ymax').text)
    # print(f'BEFORE: xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}')
    if xmin > xmax:
      xmin, xmax = xmax, xmin
    if ymin > ymax:
      ymin, ymax = ymax, ymin
    # print(f'AFTER: xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}')
    coors = [xmin, ymin, xmax, ymax]
    boxes.append(coors)
  # extract image dimensions
  width = int(root.find('.//size/width').text)
  height = int(root.find('.//size/height').text)
  depth = int(root.find('.//size/depth').text)
  return boxes, width, height, depth


def load_mask(box, h, w):
  mask = np.zeros([h, w], dtype='uint8')
  # create mask
  row_s, row_e = box[1], box[3]
  col_s, col_e = box[0], box[2]
  # print(f'col_s: {col_s}, row_s: {row_s}, col_e: {col_e}, row_e: {row_e}')
  col_s, row_s, col_e, row_e = round(col_s), round(row_s), round(col_e), round(row_e)
  mask[row_s:row_e, col_s:col_e] = 1
  return pycocotools.mask.encode(np.asarray(mask, order="F"))


class Channel(Enum):
  TRAINING = 1
  VALIDATION = 2
  TEST = 3
  ALL = 4
  DATA_DISTILLATION = 5


def get_labeled_apple_dataset(dataset_dir, channel, annotations_dir=ANNOTATIONS_TYPE):
  images_dir = dataset_dir + '/images/'
  annotations_dir = Path(dataset_dir) / annotations_dir
  all_images = os.listdir(images_dir)
  dataset_dicts = []
  train_set_size = TRAIN_SET_PERCENTAGE * TRAINABLE_SIZE
  val_set_size = VAL_SET_PERCENTAGE * TRAINABLE_SIZE
  test_set_size = TEST_SET_PERCENTAGE * TRAINABLE_SIZE
  i = 0
  all_images.sort()

  for filename in all_images:
    record = {}
    image_id = filename[:-4]
    if channel == Channel.TRAINING and int(image_id) > train_set_size:
      continue
    if channel == Channel.VALIDATION and (int(image_id) <= train_set_size or int(image_id) > train_set_size + val_set_size):
      continue
    if channel == Channel.TEST and (int(image_id) <= train_set_size + val_set_size or i >= test_set_size):
      continue
    if channel == Channel.DATA_DISTILLATION and (int(image_id) <= TRAINABLE_SIZE or int(image_id) > TOTAL_DATA_DISTILLATION_SIZE):
      continue
    i += 1
    ann_path = annotations_dir / str(image_id + str('.xml'))
    if os.path.exists(ann_path):
      record["annotation_file"] = ann_path
      record["file_name"] = os.path.join(images_dir, filename)
      record["image_id"] = image_id
      boxes, w, h, depth = extract_boxes(ann_path)
      record["width"] = w
      record["height"] = h
      record["depth"] = depth
      objs = []
      match channel:
        case Channel.TRAINING:
          size_to_print = train_set_size
        case Channel.VALIDATION:
          size_to_print = val_set_size
        case Channel.TEST:
          size_to_print = test_set_size
        case Channel.DATA_DISTILLATION:
          size_to_print = DISTILLED_DATA_SIZE
        case _:
          size_to_print = len(all_images)
      if i < size_to_print:
        print(i, "/", size_to_print, end='\r')
      else:
        logger.info(f"{i}/{size_to_print}")

      for box in boxes:
        mask = load_mask(box, h, w)
        obj = {
            "bbox": box,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": mask,
            "category_id": 0,
        }
        objs.append(obj)
      record["annotations"] = objs
      dataset_dicts.append(record)
    else:
      print(f"{ann_path} doesn't exist! Skipping {filename}")
  return dataset_dicts


def get_unlabeled_apple_dataset(dataset_dir, channel):
  all_images = os.listdir(dataset_dir)
  dataset_dicts = []
  for filename in all_images:
    image_id = filename[:-4]
    if channel == Channel.DATA_DISTILLATION and (int(image_id) <= TRAINABLE_SIZE or int(image_id) > TOTAL_DATA_DISTILLATION_SIZE):
      continue
    record = {}
    record["file_name"] = os.path.join(dataset_dir, filename)
    shape = cv2.imread(record["file_name"]).shape
    record["height"], record["width"], record["depth"] = shape
    record["image_id"] = image_id
    dataset_dicts.append(record)
  return dataset_dicts

def train(cfg):
  """
  Runs the script in training mode. It constructs a custom trainer class with validation modality.
  If cfg.SOLVER.IMS_PER_BATCH is bigger than 1, distilled dataset is used for training.
  Args:
    cfg (CfgNode): The structure that holds common configurations for all modes.
    It is adjusted according to training in the function below.
  """
  if VAL_SET_PERCENTAGE <= 0.0:
    raise Exception("Validation set percentage must be non-zero.")

  cfg.OUTPUT_DIR = "output/{}_{}".format(ANNOTATIONS_TYPE, TRAINABLE_SIZE)
  cfg.SOLVER.MAX_ITER = int(args.itr / cfg.SOLVER.IMS_PER_BATCH)
  DatasetCatalog.register(
      "apple_train", lambda: (get_labeled_apple_dataset(DATASET_DIR_1, Channel.TRAINING)))#, apply_specific_augmentation())
  MetadataCatalog.get("apple_train").set(thing_classes=["apple"])
  cache_path = os.path.join(cfg.OUTPUT_DIR, f"apple_train_coco_format.json")
  convert_to_coco_json("apple_train", cache_path)
  cfg.DATASETS.TEST = ("apple_val",)
  DatasetCatalog.register(
      "apple_val", lambda: get_labeled_apple_dataset(DATASET_DIR_1, Channel.VALIDATION))
  MetadataCatalog.get("apple_val").set(thing_classes=["apple"])

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  validation_cfg = cfg.clone()
  validation_cfg.TEST.EVAL_PERIOD = 200 #30
  cfg.SOLVER.CHECKPOINT_PERIOD = validation_cfg.TEST.EVAL_PERIOD
  validation_cfg.MODEL.DEVICE = "cuda"
  validation_cfg.SOLVER.IMS_PER_BATCH = 1
  cfg.TEST.EVAL_PERIOD = 0 # Set to 0 to disable trainer model validation
  trainer = TrainerWithValidation(cfg, validation_cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()

def eval(cfg, suffix):
  """
  Runs the script in evaluation mode. EVAL_WEIGHTS_FILE has to exist.
  Args:
    cfg (CfgNode): The structure that holds common configurations for all modes.
    It is adjusted according to evaluation in the function below.

    suffix (string): The suffix of the output directory.
  """
  # For teacher model evaluation
  cfg.OUTPUT_DIR = "output/{}_{}/".format(ANNOTATIONS_TYPE, TRAINABLE_SIZE)
  # For student model evaulation
  #cfg.OUTPUT_DIR = "../output/distill_on_{}_{}_{}_{}/evalWeights".format(ANNOTATIONS_TYPE, TRAINABLE_SIZE, TOTAL_DATA_DISTILLATION_SIZE, suffix)
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, EVAL_WEIGHTS_FILE)
  cfg.SOLVER.IMS_PER_BATCH = 1
  model = build_model(cfg)
  DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
  DatasetCatalog.register(
      "apple_test", lambda: get_labeled_apple_dataset(DATASET_DIR_1, Channel.TEST))
  MetadataCatalog.get("apple_test").set(thing_classes=["apple"])
  evaluator = COCOEvaluator(
      "apple_test", ("bbox", ), False, output_dir="./output/")
  data_loader = build_detection_test_loader(cfg, "apple_test")
  results = inference_on_dataset(model, data_loader, evaluator)
  logger.info(results)

def batch_predict(cfg):
  cfg.OUTPUT_DIR = "output/{}_{}".format(ANNOTATIONS_TYPE, TRAINABLE_SIZE)
  # path to the model we just trained
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, EVAL_WEIGHTS_FILE)
  DatasetCatalog.register(
      "apple_test", lambda: get_labeled_apple_dataset(DATASET_DIR_1, Channel.TEST))
  MetadataCatalog.get("apple_test").set(thing_classes=["apple"])
  apple_metadata = MetadataCatalog.get("apple_test")
  dataset = DatasetCatalog.get("apple_test")
  predictor = DefaultPredictor(cfg)
  for d in dataset:
    logger.info(f"Doing prediction on {d['file_name']}")
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    # Visualize prediction result
    pred_v = Visualizer(im[:, :, ::-1],
                        metadata=apple_metadata,
                        scale=1.0,
                        instance_mode=ColorMode.IMAGE_BW
                        )
    predictions = outputs["instances"].to("cpu")

    #print(f'Predicted Boxes: {predictions.pred_boxes.tensor}')
    print(f'Number of Apple that founded by model: {len(predictions)}')

    pred_v.draw_instance_predictions(predictions)
    pred_v.draw_rectangular_hull(predictions)
    pred_out = pred_v.draw_dataset_dict(d) # (d, draw_masks=False)
    cv2.imwrite(
        "detections/"+str(d["image_id"])+"_pred.png", pred_out.get_image()[:, :, ::-1])

    # print(f'd: {d["annotation_file"]}')

    boxes, w, h, depth = extract_boxes(d["annotation_file"] )
    predicted_boxes = predictions.pred_boxes.tensor
    predicted_boxes = predicted_boxes.tolist()

    correct_predictions = []

    for predicted_box in predicted_boxes:
      for solution_box in boxes:
        xmin_solution, ymin_solution, xmax_solution, ymax_solution= solution_box
        if xmin_solution > xmax_solution: xmin_solution = xmax_solution
        if ymin_solution > ymax_solution: ymin_solution = ymax_solution

        xmin_predicted, ymin_predicted, xmax_predicted, ymax_predicted = predicted_box
        if xmin_predicted > xmax_predicted: xmin_predicted = xmax_predicted
        if ymin_predicted > ymax_predicted: ymin_predicted = ymax_predicted

        dist_thres = 50
        # print(f'xmin_predicted: {xmin_predicted}, xmin_solution: {xmin_solution} // ymin_predicted: {ymin_predicted}, ymin_solution: {ymin_solution}')
        if (xmin_solution - dist_thres) <= xmin_predicted and xmin_predicted <= (xmin_solution + dist_thres) and (xmax_solution - dist_thres) <= xmax_predicted and xmax_predicted <= (xmax_solution + dist_thres):
          if (ymin_solution - dist_thres) <= ymin_predicted and ymin_predicted <= (ymin_solution + dist_thres) and (ymax_solution - dist_thres) <= ymax_predicted and ymax_predicted <= (ymax_solution + dist_thres):
            correct_predictions.append([xmin_predicted, ymin_predicted, xmax_predicted, ymax_predicted])
            # correct_predictions.append({'xmin_predicted': round(xmin_predicted), 'xmin_solution': xmin_solution,
                                     # 'ymin_predicted': round(ymin_predicted), 'ymin_solution': ymin_solution})

    correct_predictions = [[round(element) for element in sublist] for sublist in correct_predictions]
    correct_predictions = sorted(correct_predictions, key=lambda x: x[0])
    predicted_boxes = [[round(element) for element in sublist] for sublist in predicted_boxes]
    predicted_boxes = sorted(predicted_boxes, key=lambda x: x[0])
    wrong_predictions = [sublist for sublist in predicted_boxes if sublist not in correct_predictions]
    print(f'NUMBER_OF_WRONG_PREDICTIONS: {wrong_predictions}')
    draw_boxes_on_image(im, wrong_predictions)
    cv2.imwrite(
      "detections/wrong_pred/" + str(d["image_id"]) + "_wrong_prediction.png", im)

    print(f'NUMBER_OF_CORRECT_PREDICTIONS: {len(correct_predictions)}')
    print(f'NUMBER_OF_WRONG_PREDICTIONS: {len(wrong_predictions)}')
    print(f'NUMBER_OF_MODELS_PREDICTIONS: {len(predictions)}')
    print(f'NUMBER_OF_ACTUAL_BOUNDING_BOXES: {len(boxes)}')

    print(f'wrong_predictions / predictions : {round(len(wrong_predictions) / len(predictions), 2)}')
    print(f'correct_predictions / predictions : {round(len(correct_predictions) / len(predictions), 2)}')
    print(f'predictions / boxes : {round(len(predictions) / len(boxes), 2)}')
    print(f'correct_predictions / boxes : {round(len(correct_predictions) / len(boxes), 2)}')



def single_predict(cfg):
  cfg.OUTPUT_DIR = "output/{}_{}".format(ANNOTATIONS_TYPE, TRAINABLE_SIZE)
  # For student model evaulation # Benim halt yemem!!
  #cfg.OUTPUT_DIR = "../output/distill_on_{}_{}_{}_{}".format(
      #ANNOTATIONS_TYPE, TRAINABLE_SIZE, TOTAL_DATA_DISTILLATION_SIZE, "online")
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, EVAL_WEIGHTS_FILE)
  predictor = DefaultPredictor(cfg)
  MetadataCatalog.get("apple_test").set(thing_classes=["apple"])
  apple_metadata = MetadataCatalog.get("apple_test")
  im = cv2.imread(args.test_img)
  outputs = predictor(im)
  pred_v = Visualizer(im[:, :, ::-1],
                      metadata=apple_metadata,
                      scale=2.0,
                      # remove the colors of unsegmented pixels. This option is only available for segmentation models
                      instance_mode=ColorMode.IMAGE_BW
                      )
  predictions = outputs["instances"].to("cpu")
  print(len(predictions))
  #print(f'predictions: {predictions}')
  pred_v.draw_instance_predictions(predictions)
  pred_out = pred_v.draw_rectangular_hull(predictions)
  cv2.imwrite(ANNOTATIONS_TYPE.split('_')[-1] + "_output_"+ args.test_img.split('.')[0]+"_2.png", pred_out.get_image()[:, :, ::-1])

def tune(cfg, all_images):
  """
  Output of this function is used to tune hyperparameters according to the negated validation loss on a subset of dataset.
  """
  TRAIN_SET_PERCENTAGE = 0.3
  # Do not remove this line as VAL_SET_PERCENTAGE is used in get_labeled_apple_dataset()
  VAL_SET_PERCENTAGE = 0.05
  cfg.DATASETS.TEST = ("apple_val",)
  DatasetCatalog.register(
      "apple_val", lambda: get_labeled_apple_dataset(DATASET_DIR_1, Channel.VALIDATION))
  MetadataCatalog.get("apple_val").set(thing_classes=["apple"])
  cfg.SOLVER.MAX_ITER = int(
      len(all_images)*TRAIN_SET_PERCENTAGE*NUM_VAL_EPOCH)
  cfg.TEST.EVAL_PERIOD = len(all_images)*TRAIN_SET_PERCENTAGE
  DatasetCatalog.register(
      "apple_train", lambda: get_labeled_apple_dataset(DATASET_DIR_1, Channel.TRAINING))
  MetadataCatalog.get("apple_train").set(thing_classes=["apple"])

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = TrainerWithValidation(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()
  print(-trainer.model.latest_val_loss)

def save_fpn_layers(cfg):
  def save_feature(output, key):
      feature = output[key][0][0].cpu().detach().numpy()
      feature = (255*(feature - np.min(feature)) /
                 np.ptp(feature)).astype(np.uint8)
      cv2.imwrite(key+".png", feature)
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, EVAL_WEIGHTS_FILE)
  model = build_model(cfg)
  DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
  inp_im = cv2.imread("test.png")
  inp_im = cv2.resize(inp_im, (1280, 1280))
  inp = np.zeros([1, 3, 1280, 1280], dtype=np.float32)
  inp[0] = np.reshape(inp_im, [3, 1280, 1280])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  inp = torch.from_numpy(inp).float().to(device)
  output = model.backbone(inp)
  save_feature(output, "p2")
  save_feature(output, "p3")
  save_feature(output, "p4")
  save_feature(output, "p5")

def distill_data(cfg, debug_data_distillation = False):
  """
  Conducts data distillation on the unlabeled dataset. 
  
  Args:
    cfg (CfgNode): The structure that holds common configurations for all modes.
    It is adjusted according to data distillation in the function below.

    debug_data_distillation (bool): Saves the overlayed image of auto-generated annotation to the disk.  
  """
  cfg.SOLVER.IMS_PER_BATCH = 1
  cfg.OUTPUT_DIR = "output/{}_{}".format(ANNOTATIONS_TYPE, TRAINABLE_SIZE)
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, EVAL_WEIGHTS_FILE)

  unlabeled_dataset_dir = DATASET_DIR_1
  DatasetCatalog.register(
      "apple_data_distillation", lambda: get_unlabeled_apple_dataset(unlabeled_dataset_dir + '/images', Channel.DATA_DISTILLATION))
  MetadataCatalog.get("apple_data_distillation").set(thing_classes=["apple"])
  dataset = DatasetCatalog.get("apple_data_distillation")

  logger.info(f"Data distillation on {len(dataset)} images")
  results = multi_transform_inference(cfg, dataset)

  if debug_data_distillation:
    apple_metadata = MetadataCatalog.get("apple_data_distillation")

  for d in dataset:
    img_id = int(d["image_id"])
    if img_id in results.keys():
      target_annotation_file = unlabeled_dataset_dir + \
          '/auto_{}_{}_{}/'.format(ANNOTATIONS_TYPE, TRAINABLE_SIZE,
                                    TOTAL_DATA_DISTILLATION_SIZE) + d["image_id"] + '.xml'
      print(f"Creating annotation file for {img_id}.")
      create_annotation_file(target_annotation_file, d, results[img_id])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # folder_path = unlabeled_dataset_dir + '/auto_{}_{}_{}/'.format(ANNOTATIONS_TYPE, TRAINABLE_SIZE, TOTAL_DATA_DISTILLATION_SIZE)
  # files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
  # num_files = len(files)
  # print(f'The number of files in the folder is: {num_files}')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      if debug_data_distillation:
        im = cv2.imread(d["file_name"])
        pred_v = Visualizer(im[:, :, ::-1],
                            metadata=apple_metadata,
                            scale=1.0,
                            instance_mode=ColorMode.IMAGE_BW
                            )
        pred_out = pred_v.overlay_instances(
            masks=None,
            boxes=results[img_id],
            labels=["apple"]*len(results[img_id]),
            keypoints=None,
            assigned_colors=None,
            alpha=1.0,
        )
        cv2.imwrite(
            "data_distillation/filtered/"+d["image_id"]+"_pred.png", pred_out.get_image()[:, :, ::-1])
        print(f'CREATED: data_distillation/filtered/{d["image_id"]}_pred.png')

def retrain_with_data_distillation(cfg, suffix):
  """
  Conducts data distillation on the unlabeled dataset. 
  
  Args:
    cfg (CfgNode): The structure that holds common configurations for all modes.
    It is adjusted according to retraining with data distillation in the function below.

    suffix (string): The suffix of the output directory.
  """
  cfg.OUTPUT_DIR = "../output/distill_on_{}_{}_{}_{}".format(
      ANNOTATIONS_TYPE, TRAINABLE_SIZE, TOTAL_DATA_DISTILLATION_SIZE, suffix)
  print(f"Output dir: {cfg.OUTPUT_DIR}")

  cfg.SOLVER.MAX_ITER = int(args.itr * 1.25)
  print(f"Total number of training iterations is {cfg.SOLVER.MAX_ITER}")

  cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER*0.7), int(cfg.SOLVER.MAX_ITER*0.9))
  cfg.SOLVER.GAMMA = 0.1

  cfg.DATASETS.TRAIN = ("manual_annotation","auto_annotation")

  DatasetCatalog.register(
      "manual_annotation", lambda: get_labeled_apple_dataset(
          DATASET_DIR_1, Channel.TRAINING))
  # From other dataset: get_labeled_apple_dataset(DATASET_DIR_2, Channel.ALL, "auto_annotations_unfiltered")

  DatasetCatalog.register("auto_annotation", lambda: get_labeled_apple_dataset(
      DATASET_DIR_1, Channel.DATA_DISTILLATION, "auto_" + ANNOTATIONS_TYPE + "_" + str(TRAINABLE_SIZE) + "_" + str(TOTAL_DATA_DISTILLATION_SIZE)))

  MetadataCatalog.get("manual_annotation").set(thing_classes=["apple"])
  MetadataCatalog.get("manual_annotation").set(size=int(TRAINABLE_SIZE*TRAIN_SET_PERCENTAGE))

  MetadataCatalog.get("auto_annotation").set(thing_classes=["apple"])
  MetadataCatalog.get("auto_annotation").set(size=DISTILLED_DATA_SIZE)

  cfg.DATASETS.TEST = ("apple_val",)
  DatasetCatalog.register(
      "apple_val", lambda: get_labeled_apple_dataset(DATASET_DIR_1, Channel.VALIDATION))
  MetadataCatalog.get("apple_val").set(thing_classes=["apple"])

  validation_cfg = cfg.clone()
  validation_cfg.TEST.EVAL_PERIOD = int(cfg.SOLVER.MAX_ITER/10)
  cfg.SOLVER.CHECKPOINT_PERIOD = validation_cfg.TEST.EVAL_PERIOD
  validation_cfg.MODEL.DEVICE = "cuda"
  validation_cfg.SOLVER.IMS_PER_BATCH = 1
  cfg.TEST.EVAL_PERIOD = 0 # Set to 0 to disable trainer model validation
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

  trainer = TrainerWithValidation(cfg, validation_cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()

def run(args):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(
      BASE_MODEL_FILE))
  cfg.DATASETS.TRAIN = ("apple_train",)
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
      BASE_MODEL_FILE)  # Let training initialize from model zoo
  if args.multi_batch:
    cfg.SOLVER.IMS_PER_BATCH = 8
  else:
    cfg.SOLVER.IMS_PER_BATCH = 1
  cfg.SOLVER.BASE_LR = float(args.learning_rate)  # pick a good LR
  all_images = os.listdir(Path(DATASET_DIR_1) / 'images')

  cfg.SOLVER.STEPS = []
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = float(args.rpn_bbox_loss_weight)
  cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = float(
      args.roi_bbox_loss_weight)
  cfg.MODEL.DEVICE = args.device
  cfg.INPUT.MASK_FORMAT = "bitmask"

  if args.mode == TRAINING_MODE:
    train(cfg)
  elif args.mode == EVALUATION_MODE:
    suffix = "online"
    if args.multi_batch:
      suffix = "multi_batch"
    eval(cfg, suffix)
  elif args.mode == BATCH_PREDICTION_MODE:
    batch_predict(cfg)
  elif args.mode == PREDICTION_MODE:
    single_predict(cfg)
  elif args.mode == TUNING_MODE:
    tune(cfg, all_images)
  elif args.mode == FPN_LAYERS_MODE:
    save_fpn_layers(cfg)
  elif args.mode == DATA_DISTILLATION_MODE:
    distill_data(cfg, args.debug_data_distillation)
  elif args.mode == RETRAINING_MODE:
    suffix = "online"
    if args.multi_batch:
      suffix = "multi_batch"
    retrain_with_data_distillation(cfg, suffix) #retrain_with_data_distillation(suffix) Benim halt yemem
  elif args.mode == ANNOTATION_MERGE_MODE:
    autolabeled_dataset_dicts = get_labeled_apple_dataset(DATASET_DIR_2, Channel.ALL, "auto_annotations_filtered")
    merge_illumination_annotations(autolabeled_dataset_dicts)
  else:
    pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Apple detector parser')
  parser.add_argument("-m", "--mode", type=str,
                      default=TRAINING_MODE, choices=[TRAINING_MODE, EVALUATION_MODE, BATCH_PREDICTION_MODE,
                                                      PREDICTION_MODE, TUNING_MODE, FPN_LAYERS_MODE, DATA_DISTILLATION_MODE,
                                                      RETRAINING_MODE, ANNOTATION_MERGE_MODE], help="Mode of script")
  parser.add_argument("-d", "--device", type=str,
                      default="cuda", help="Device")
  parser.add_argument("-t", "--test-img", type=str)
  parser.add_argument("-l", "--learning-rate", type=float,
                      default=0.003528481432413011)
  parser.add_argument("-b", "--rpn-bbox-loss-weight",
                      type=float, default=0.1628283760409546)
  parser.add_argument("-c", "--roi-bbox-loss-weight",
                      type=float, default=0.05016807725153617)
  parser.add_argument("-i", "--itr", type=int, default=2000)
  parser.add_argument("--multi-batch", default=False, action='store_true')
  parser.add_argument("--debug-data-distillation", default=False, action='store_true')
  args, _ = parser.parse_known_args()
  run(args)

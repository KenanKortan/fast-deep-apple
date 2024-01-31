import numpy
import numpy as np
import torch
from enum import Enum
import cv2
from collections import defaultdict
import time
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from detectron2.structures.boxes import Boxes, matched_pairwise_iou
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.modeling import build_model
import pickle
import os
from from_root import from_root
from PIL import Image

DATA_DISTILLATION_SCALES = np.arange(400, 1300, 100)
COLOR_SCOPE = np.arange(0.2, 2.1, 0.3)
DATA_DISTILLATION_PREDICTION_THRESH = 0.85
MAX_PIXEL_DIST_THRESH = 30
MIN_PIXEL_DIST_THRESH = 5

TRAINABLE_SIZE = 240
DISTILLED_DATA_SIZE = 480

#UNLABELED_DATASET_DIR = str(from_root("datasets/ApplesAnnotated-1374")) # BEnim Halt yemem!!!
UNLABELED_DATASET_DIR = str(from_root("datasets/ApplesAnnotated-800"))

class Transformations(Enum):
  NO_OP = 1
  FLIP = 2
  # Custome that I did
  ROTATION = 3
  BRIGHTNESS = 4
  CONTRAST = 5
  SATURATION = 6
  LIGHTNING = 7

class BBox:
  def __init__(self, ind, bbox):
    self.ind = ind
    self.bbox = bbox
  def __str__(self):
    return str(self.bbox)


def get_center(bbox):
  bbox_c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
  return bbox_c


"""
  Returns:
      List(BBox): Pair of bboxes
"""


def compute_min_bbox_pair(bbox_predictions):
  boxes = Boxes(torch.FloatTensor(bbox_predictions))
  min_score = 1512215121715
  result_pair = []
  for i in range(0, len(bbox_predictions)):
    b1 = bbox_predictions[i]
    b1c = get_center(b1)
    for j in range(i+1, len(bbox_predictions)):
      b2 = bbox_predictions[j]
      b2c = get_center(b2)
      dist = np.linalg.norm(b2c-b1c)
      if dist < MIN_PIXEL_DIST_THRESH:
        return [BBox(i, b1), BBox(j, b2)]
      else:
        pair = Boxes(torch.FloatTensor([b1, b2]))
        iou = matched_pairwise_iou(pair, pair)[0]
        if iou > 0.5:
          if dist < min_score:
            min_score = dist
            result_pair = [i, j]
  if min_score <= MAX_PIXEL_DIST_THRESH:
    return [BBox(result_pair[0], bbox_predictions[result_pair[0]]), BBox(result_pair[1], bbox_predictions[result_pair[1]])]
  else:
    return None


def merge_boxes(bbox_predictions, bbox_pair):
  b1 = bbox_pair[0]
  b2 = bbox_pair[1]
  bbox_predictions = np.delete(bbox_predictions, [b1.ind, b2.ind], 0).tolist()
  merged_bbox = ((np.array(b1.bbox) + np.array(b2.bbox))/2.0).tolist()
  bbox_predictions.append(merged_bbox)
  return bbox_predictions


def remove_duplicates(bbox_predictions):
  indices_to_delete = []
  for i in range(0, len(bbox_predictions)):
    if i in indices_to_delete:
      continue
    b1 = bbox_predictions[i]
    b1c = get_center(b1)
    for j in range(i+1, len(bbox_predictions)):
      b2 = bbox_predictions[j]
      b2c = get_center(b2)
      dist = np.linalg.norm(b2c-b1c)
      if dist < 2:
        indices_to_delete.append(j)
  bbox_predictions = np.delete(bbox_predictions, indices_to_delete, 0).tolist()
  return bbox_predictions

def bbox_combine(all_bbox_predictions):
  i = 1
  for img_id, bbox_predictions in list(all_bbox_predictions.items()):
    #if img_id < TRAINABLE_SIZE or img_id > TRAINABLE_SIZE + DISTILLED_DATA_SIZE:
    #  all_bbox_predictions.pop(img_id)
    #  continue
    print(f"{i}- In image {img_id}")
    i+=1
    bbox_predictions = remove_duplicates(bbox_predictions)
    print(f'Duplicated coordinates removed')
    while True:
      bbox_pair = compute_min_bbox_pair(bbox_predictions)
      print(f'Min bbox pair computed')
      if bbox_pair is None:
        break
      else:
        bbox_predictions = merge_boxes(bbox_predictions, bbox_pair)
        print(f'Boxes merged')
      all_bbox_predictions[img_id] = bbox_predictions
  print(f'Ready to go')
  return all_bbox_predictions


class TransformPredictor:
  def __init__(self, cfg, augs):
    self.cfg = cfg.clone()
    self.model = build_model(self.cfg)
    self.model.eval()
    if len(cfg.DATASETS.TEST):
      self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    checkpointer = DetectionCheckpointer(self.model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    self.augs = augs
    self.input_format = cfg.INPUT.FORMAT
    assert self.input_format in ["RGB", "BGR"], self.input_format
    self.augmented_image = None

  def __call__(self, original_image):
    with torch.no_grad():
      if self.input_format == "RGB":
        original_image = original_image[:, :, ::-1]

      # Apply augmentations and store the augmented image
      augmented_image, transforms = T.apply_transform_gens(self.augs, original_image)
      print(f'SELF_AUGS: {self.augs}')
      self.augmented_image = augmented_image
      height, width = augmented_image.shape[:2]

      image = torch.as_tensor(augmented_image.astype("float32").transpose(2, 0, 1))
      inputs = {"image": image, "height": height, "width": width}
      predictions = self.model([inputs])[0]
      return predictions

  def get_augmented_image(self):
    return self.augmented_image

#~!Solution candidate 2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calculate_original_bbox_coordinates(rotated_bbox, rotation_angle, rotation_center=None):
    """
    Calculate the original bounding box coordinates using rotated bounding box coordinates.

    Args:
        rotated_bbox (list): Rotated bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        rotation_angle (float): Rotation angle in degrees.
        rotation_center (tuple): Rotation center coordinates in the format (x, y). Default is None.

    Returns:
        list: Original bounding box coordinates in the format [x_min, y_min, x_max, y_max].
    """
    # Convert bbox coordinates to corner points
    rotated_corners = np.array([
        [rotated_bbox[0], rotated_bbox[1]],
        [rotated_bbox[2], rotated_bbox[1]],
        [rotated_bbox[2], rotated_bbox[3]],
        [rotated_bbox[0], rotated_bbox[3]],
    ])

    # If rotation center is not provided, use the center of the bounding box
    if rotation_center is None:
        rotation_center = ((rotated_bbox[0] + rotated_bbox[2]) / 2, (rotated_bbox[1] + rotated_bbox[3]) / 2)

    # Apply inverse rotation to corner points
    inverse_rotation_matrix = np.array([[np.cos(np.radians(-rotation_angle)), -np.sin(np.radians(-rotation_angle))],
                                        [np.sin(np.radians(-rotation_angle)), np.cos(np.radians(-rotation_angle))]])
    original_corners = np.dot(rotated_corners - rotation_center, inverse_rotation_matrix.T) + rotation_center

    # Calculate new bounding box coordinates
    original_bbox = [
        np.min(original_corners[:, 0]),
        np.min(original_corners[:, 1]),
        np.max(original_corners[:, 0]),
        np.max(original_corners[:, 1]),
    ]

    return original_bbox
#~!Solution candidate1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def rotate_point(point, angle, center):
    """
    Rotate a point around a center by a given angle.

    Args:
        point (tuple): Coordinates of the point (x, y).
        angle (float): Rotation angle in degrees.
        center (tuple): Coordinates of the center (cx, cy).

    Returns:
        tuple: Rotated coordinates of the point.
    """
    x, y = point
    cx, cy = center
    angle_rad = np.radians(angle)
    new_x = (x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad) + cx
    new_y = (x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad) + cy
    return new_x, new_y

def rotate_bounding_box(original_bbox, angle, center):
    """
    Calculate the original bounding box coordinates after rotation.

    Args:
        original_bbox (list): Coordinates of the rotated bounding box in the format [x_min, y_min, x_max, y_max].
        angle (float): Rotation angle in degrees.
        center (tuple): Coordinates of the center (cx, cy).

    Returns:
        list: Coordinates of the original bounding box in the format [x_min, y_min, x_max, y_max].
    """
    # Convert to format suitable for rotation transformation (list of (x, y))
    rotated_points = [
        (original_bbox[0], original_bbox[1]),
        (original_bbox[2], original_bbox[1]),
        (original_bbox[2], original_bbox[3]),
        (original_bbox[0], original_bbox[3])
    ]

    # Rotate each corner point
    rotated_points = [rotate_point(point, angle, center) for point in rotated_points]

    # Calculate the bounding box of the rotated points
    x_coords, y_coords = zip(*rotated_points)
    x_min, y_min, x_max, y_max = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    return [x_min, y_min, x_max, y_max]
#~!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_boxes_on_image(image, boxes, filename, color=(0, 255, 0), thickness=2):
  for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    # print(f'Draw Boxes On Image => x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
  cv2.imwrite(filename, image)

def apply_flip_transformation(b, scale):
    flip_line = scale / 2.0
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
    TF = np.matrix([[1, 0, flip_line], [0, 1, 0], [0, 0, 1]])
    inv_TF = np.matrix([[1, 0, -flip_line], [0, 1, 0], [0, 0, 1]])
    F = np.matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x1, y1, _ = np.squeeze(np.asarray(TF * F * inv_TF * np.matrix([x1, y1, 1]).T))
    x2, y2, _ = np.squeeze(np.asarray(TF * F * inv_TF * np.matrix([x2, y2, 1]).T))
    return np.array([x1, y1, x2, y2])

def reverse_scaling(b, scale, width, height):
    scaled_x1, scaled_y1, scaled_x2, scaled_y2 = b[0], b[1], b[2], b[3]
    original_x1 = (scaled_x1 / (scale / width))
    original_y1 = (scaled_y1 / (scale / height))
    original_x2 = (scaled_x2 / (scale / width))
    original_y2 = (scaled_y2 / (scale / height))
    return np.array([original_x1, original_y1, original_x2, original_y2])

def process_image_data(cfg, dataset, augs, results, coord_dict, scale, method, val=None, rotated_box=None, rotation_angle= None):

    method_str = "FLIP" if method == Transformations.FLIP else \
        "ROTATION" if method == Transformations.ROTATION else \
        "BRIGHTNESS" if method == Transformations.BRIGHTNESS else \
        "CONTRAST" if method == Transformations.CONTRAST else \
        "SATURATION" if method == Transformations.SATURATION else "NORMAL"

    total_time = 0.0
    predictor = TransformPredictor(cfg, augs)
    start = time.time()

    for d in dataset:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        augmented_image = predictor.get_augmented_image()
        '''
        # --------------------------- Save the Augmented image without bounding boxes ----------------------------------
        if method_str in ["BRIGHTNESS", "CONTRAST", "SATURATION"]:
            cv2.imwrite(f'detections/drawed/undrawed/{scale}_{method_str}_{val}_{int(d["image_id"])}.png', augmented_image)
        else:
            cv2.imwrite(f'detections/drawed/undrawed/{scale}_{method_str}_{int(d["image_id"])}.png', augmented_image)
        #---------------------------------------------------------------------------------------------------------------
        '''
        wi, he, ch = augmented_image.shape

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.tolist()
        '''
        # -------------------------- Save the Augmented image with Teacher model's predicted bounding boxes ------------
        augmented_image = np.array(augmented_image)
        if method_str in ["BRIGHTNESS", "CONTRAST", "SATURATION"]:
            draw_boxes_on_image(augmented_image, boxes, filename= f'detections/drawed/predicted/{scale}_{method_str}_{val}_{int(d["image_id"])}.png')
        else:
            draw_boxes_on_image(augmented_image, boxes, filename= f'detections/drawed/predicted/{scale}_{method_str}_{int(d["image_id"])}.png')
        # --------------------------------------------------------------------------------------------------------------
        '''
        for b, s in zip(boxes, scores):
            if s < DATA_DISTILLATION_PREDICTION_THRESH:
                continue

            if method == Transformations.FLIP:
                b = apply_flip_transformation(b, scale)
            '''
            if method == Transformations.ROTATION:
                if rotation_angle != None:
                    # xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
                    # xmin,ymax-------xmax,ymax
                    # |                   |
                    # |                   |
                    # |                   |
                    # xmin,ymin-------xmax,ymin
                    print(f'BEFORE (UNSHAPE_B): {b}')
                    b = np.asarray([
                        [b[0], b[1]],
                        [b[2], b[1]],
                        [b[2], b[3]],
                        [b[0], b[3]]
                    ])
                    print(f'AFTER (SHAPED_B): {b}')
                    b= T.RotationTransform(h=scale, w=scale, angle=360-rotation_angle).apply_coords(b)
                    print(f'BEFORE (ROT_TO_NORM): {b}')
                    b= np.asarray([b[0][0],b[0][1],b[2][0],b[2][1]])
                    print(f'AFTER (ROT_TO_NORM): {b}')
                    rotated_box[int(scale)].append(b)
                
                # Example usage 1:
                rotation_angle = 330  # Replace with the actual rotation angle used
                rotation_center = (scale/2, scale/2)  # Replace with the actual rotation center used
                #rotation_center = (wi / 2, he / 2)
                #rotation_center = (d["width"] / 2, d["height"] / 2)
                b = numpy.round(rotate_bounding_box(b, rotation_angle, rotation_center), 8)
                rotated_box[int(scale)].append(b)
                # print("Original (NOT ROTATED) Bounding Box:", b)
                
                # Example usage 2:
                # rotation_center = (scale/2, scale/2)
                #rotation_center = (wi / 2, he / 2)
                # rotation_center = (he / 2, wi / 2)
                rotation_angle = 30  # rotation angle
                # Use the function to calculate original bounding box coordinates
                #b = numpy.round(calculate_original_bbox_coordinates(b, rotation_angle, rotation_center), 8)
                b = calculate_original_bbox_coordinates(b, rotation_angle) # ,rotation_center
                rotated_box[int(scale)].append(b)
                '''
            b = reverse_scaling(b, scale, int(d["width"]), int(d["height"]))

            if b[2] < b[0]:
                b[0], b[2] = b[2], b[0]
            if b[3] < b[1]:
                b[1], b[3] = b[3], b[1]

            results[int(d["image_id"])].append(b)
            '''
            #------------------------ Dictionary created for saving all bounding boxes just for display-----------------
            if int(d["image_id"]) not in coord_dict:
                coord_dict[int(d["image_id"])] = {}
            if scale not in coord_dict[int(d["image_id"])]:
                coord_dict[int(d["image_id"])][scale] = {}
            if method_str not in coord_dict[int(d["image_id"])][scale]:
                coord_dict[int(d["image_id"])][scale][method_str] = []
            coord_dict[int(d["image_id"])][scale][method_str].append(b)
            # ----------------------------------------------------------------------------------------------------------
            '''
    end = time.time()
    total_time += end - start
    print(f"Method: {method_str} - Scale: {scale} took {end - start} seconds over the unlabeled dataset."
          f"Total number of images that have valid detections: {len(results.keys())}!")
    return total_time

def final_prediction_pass(cfg, dataset, coord_dict, results, total_time, rotated_box=None):
    print("Doing final prediction pass with default predictor..")

    '''
    if rotated_box != None :
        if rotated_box != {}:
            print(f'ROTATED BOX IS NOT EMPTY')
            for scale in DATA_DISTILLATION_SCALES:
                scaled_im = cv2.imread(filename=f'detections/drawed/undrawed/{scale}_NORMAL_241.png')
                draw_boxes_on_image(scaled_im, rotated_box[int(scale)], filename= f'detections/drawed/aug/{scale}_ROT_TO_NORM_241.png')
    '''
    '''
    for image_id, scale_dict in coord_dict.items():
        for scale_val, method_dict in scale_dict.items():
            for method_name, val_arr in method_dict.items():
                for d in dataset:
                    org_im = cv2.imread(d["file_name"])
                    draw_boxes_on_image(org_im, val_arr, filename= f'detections/drawed/{image_id}_{scale_val}_{method_name}.png')
    '''
    predictor = DefaultPredictor(cfg)

    for d in dataset:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        # draw_boxes_on_image(im, boxes, f'detections/drawed/DefaultPredictor_{d["image_id"]}.png')
        scores = instances.scores.tolist()

        for b, s in zip(boxes, scores):
            if s < DATA_DISTILLATION_PREDICTION_THRESH:
                continue
            results[int(d["image_id"])].append(b)

    print(f"Total elapsed time: {total_time}")

    with open("box_predictions.pkl", "wb") as f:
        pickle.dump(results, f)

    bbox_combine(results)

def multi_transform_inference(cfg, dataset):
    results = defaultdict(list)
    total_time = 0.0
    coord_dict = defaultdict(dict)
    rotated_box = defaultdict(list)

    for scale in DATA_DISTILLATION_SCALES:
        for method in [Transformations.NO_OP, Transformations.FLIP, Transformations.BRIGHTNESS,
                       Transformations.CONTRAST, Transformations.SATURATION]: #,Transformations.ROTATION]:
            # augs = []
            augs = [T.Resize((scale, scale))]

            if method == Transformations.NO_OP:
                process_image_data(cfg, dataset, augs, results, coord_dict, scale, method)

            if method == Transformations.FLIP:
                augs.append(T.RandomFlip(prob=1.0, horizontal=True))
                process_image_data(cfg, dataset, augs, results, coord_dict, scale, method)

            if method == Transformations.BRIGHTNESS:
                for val in COLOR_SCOPE:
                    val = round(val, 2)
                    augs = [T.Resize((scale, scale))] # Reset the aug list`
                    augs.append(T.RandomBrightness(intensity_min=val, intensity_max=val))
                    process_image_data(cfg, dataset, augs, results, coord_dict,scale, method, val= val)

            if method == Transformations.CONTRAST:
                for val in COLOR_SCOPE:
                    val = round(val, 2)
                    augs = [T.Resize((scale, scale))]
                    augs.append(T.RandomContrast(intensity_min=val, intensity_max=val))
                    process_image_data(cfg, dataset, augs, results, coord_dict,scale, method, val= val)

            if method == Transformations.SATURATION:
                for val in COLOR_SCOPE:
                    val = round(val, 2)
                    augs = [T.Resize((scale, scale))]
                    augs.append(T.RandomSaturation(intensity_min=val, intensity_max=val))
                    process_image_data(cfg, dataset, augs, results, coord_dict, scale, method, val= val)

            if method == Transformations.SATURATION:
                for val in COLOR_SCOPE:
                    val = round(val, 2)
                    augs.append(T.RandomSaturation(intensity_min=val, intensity_max=val))
                    process_image_data(cfg, dataset, augs, results, coord_dict, scale, method, val)
            '''
            if method == Transformations.ROTATION:
                rotation_angle= 30
                augs.append(T.RotationTransform(h=scale, w=scale, angle=rotation_angle)) #T.RandomRotation([30, 30]))
                process_image_data(cfg, dataset, augs, results, coord_dict, scale, method,rotated_box= rotated_box, rotation_angle=rotation_angle)
            '''

            #if method == Transformations.EXTENT:
                #rotation_angle= 30
                #augs.append(T.ExtentTransform(src_rect=)) #T.RandomRotation([30, 30]))
                #process_image_data(cfg, dataset, augs, results, coord_dict, scale, method,rotated_box= rotated_box, rotation_angle=rotation_angle)


    final_prediction_pass(cfg, dataset, coord_dict, results, total_time,rotated_box= rotated_box)

    return results


def merge_illumination_annotations(autolabeled_dataset_dicts):
  def find(f, seq):
    """Return first item in sequence where f(item) == True."""
    for item in seq:
      if f(item):
        return item
    return None

  class AnnotationBatch:
    def __init__(self):
      self.expected_img_ids = []
      self.actual_img_ids = []
      self.bbox_annotations = []

    def __str__(self):
      return f'Expected_img_ids: {self.expected_img_ids} | Actual_img_ids: {self.actual_img_ids} | BBox annotations: {self.bbox_annotations}'
  autolabeled_image_cnt = 20 #len(os.listdir(Path(UNLABELED_DATASET_DIR) / 'images'))
  num_illumination_batches = int(autolabeled_image_cnt / 6)
  batches = []
  for i in range(0, num_illumination_batches):
    batch = AnnotationBatch()
    for j in range(1, 7):
      img_id = i*6+j
      batch.expected_img_ids.append(img_id)
      data = find(lambda d: int(d['image_id']) ==
                  img_id, autolabeled_dataset_dicts)
      if data is None:
        continue
      batch.actual_img_ids.append(img_id)
      for annot in data['annotations']:
        batch.bbox_annotations.append(np.array(annot['bbox']))
    batches.append(batch)
  for batch in batches:
    while True:
      bbox_pair = compute_min_bbox_pair(batch.bbox_annotations)
      if bbox_pair is None:
        break
      else:
        batch.bbox_annotations = merge_boxes(
            batch.bbox_annotations, bbox_pair)
    # any would do
    if len(batch.actual_img_ids) > 0:
      img_id = batch.actual_img_ids[-1]
      data = find(lambda d: int(d['image_id']) ==
                  img_id, autolabeled_dataset_dicts)
      target_annotation_file = UNLABELED_DATASET_DIR + \
          '/annotations/' + str(img_id) + '.xml'
      create_annotation_file(target_annotation_file,
                            data, batch.bbox_annotations)


def create_annotation_file(filepath, data, boxes):
  annotation_n = Element('annotation')
  size_n = SubElement(annotation_n, 'size')
  width_n = SubElement(size_n, 'width')
  width_n.text = str(data["width"])
  height_n = SubElement(size_n, 'height')
  height_n.text = str(data["height"])
  depth_n = SubElement(size_n, 'depth')
  depth_n.text = str(data["depth"])
  for bbox in boxes:
    object_n = SubElement(annotation_n, 'object')
    name_n = SubElement(object_n, 'name')
    name_n.text = "Apple"
    bndbox_n = SubElement(object_n, 'bndbox')
    xmin_n = SubElement(bndbox_n, 'xmin')
    xmin_n.text = str(int(bbox[0]))
    ymin_n = SubElement(bndbox_n, 'ymin')
    ymin_n.text = str(int(bbox[1]))
    xmax_n = SubElement(bndbox_n, 'xmax')
    xmax_n.text = str(int(bbox[2]))
    ymax_n = SubElement(bndbox_n, 'ymax')
    ymax_n.text = str(int(bbox[3]))
  xmlstr = minidom.parseString(
      tostring(annotation_n)).toprettyxml(indent="   ")
  def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
  ensure_dir(filepath)
  with open(filepath, "w") as f:
    f.write(xmlstr)




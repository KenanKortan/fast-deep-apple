from xml.etree import ElementTree
import os
import csv
from pathlib import Path
from from_root import from_root
LABELED_DATASET_DIR =  str(from_root("datasets/ApplesAnnotated-800"))

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
    if xmin > xmax:
      xmin, xmax = xmax, xmin
    if ymin > ymax:
      ymin, ymax = ymax, ymin
    coors = [xmin, ymin, xmax, ymax]
    boxes.append(coors)
  # extract image dimensions
  width = int(root.find('.//size/width').text)
  height = int(root.find('.//size/height').text)
  depth = int(root.find('.//size/depth').text)
  return boxes, width, height, depth



if __name__=='__main__':
  images_dir = LABELED_DATASET_DIR + '/images/'
  annotations_dir = LABELED_DATASET_DIR + '/annotations_unfiltered/'
  all_images = os.listdir(images_dir)
  all_images.sort()
  with open('annotations_unfiltered.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "file_size", "file_attributes", "region_count", "region_id", "region_shape_attributes", "region_attributes"])
    for filename in all_images:
      image_id = filename[:-4]
      ann_path = annotations_dir + image_id + '.xml'
      if os.path.exists(ann_path):
        boxes, _, _, _ = extract_boxes(ann_path)
        sz = Path(os.path.join(images_dir, filename)).stat().st_size
        for i in range(0,len(boxes)):
          box_width = boxes[i][2] - boxes[i][0]
          box_height = boxes[i][3] - boxes[i][1]
          region_shape_attributes = "{\"name\":\"rect\",\"x\":" + str(boxes[i][0]) + ", \"y\":" + str(
              boxes[i][1]) + ", \"width\":" + str(box_width) + ", \"height\":"+str(box_height) + "}"
          region_attributes = "{\"apple\":\"apple\"}"
          writer.writerow([filename, sz, "{ }", len(boxes), i, region_shape_attributes, region_attributes])

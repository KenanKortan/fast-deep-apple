import csv
from collections import defaultdict
import re
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from from_root import from_root
LABELED_DATASET_DIR =  str(from_root("datasets/ApplesAnnotated-800"))

def create_annotation_file(filepath, width, height, depth, boxes):
  annotation_n = Element('annotation')
  size_n = SubElement(annotation_n, 'size')
  width_n = SubElement(size_n, 'width')
  width_n.text = str(width)
  height_n = SubElement(size_n, 'height')
  height_n.text = str(height)
  depth_n = SubElement(size_n, 'depth')
  depth_n.text = str(depth)
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
  with open(filepath, "w") as f:
    f.write(xmlstr)

if __name__=='__main__':
  new_annotations_dir = LABELED_DATASET_DIR + '/annotations_unfiltered'
  with open('apples_800_annotation.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    annots = defaultdict(list)
    for row in csv_reader:
      if line_count == 0:
        line_count+=1
        continue
      filename = row[0]
      image_id = filename[:-4]
      tokens = row[5].split(',')
      x_min= int(re.search(r'\d+', tokens[1]).group())
      y_min= int(re.search(r'\d+', tokens[2]).group())
      x_max= x_min +  int(re.search(r'\d+', tokens[3]).group())
      y_max= y_min +  int(re.search(r'\d+', tokens[4]).group())
      annots[image_id].append([x_min, y_min, x_max, y_max])
      line_count+=1
    for img, boxes in annots.items():
      target_annotation_file = new_annotations_dir + '/' + str(img) + '.xml'
      create_annotation_file(target_annotation_file, 1920, 1080, 3, boxes)


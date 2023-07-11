from dataset.Augmentation import *
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import json
import os

class RefCOCOg(Dataset):
  '''
  RefCOCOg dataset parser

  Args:
      - refcocog_path: path to the directory containing the annotations and
        images folders of the refcocog dataset
      - split: when set to "test" this class will load the test part of the
          dataset, when "val" it will load the validation set, otherwise it
          loads the "train" set
      - size: used to extract only a fraction of the whole dataset. This number
        specifies the number of samples to include
      - transformations: when set to True enables transformations
      - device: either "cpu" or "cuda"
  '''
  def __init__(self, refcocog_path, split, size=None, transformations=False, device=None):
    self.split = split
    self.img_dir = os.path.join(refcocog_path, "images")
    self.annotations_dir = os.path.join(refcocog_path, "annotations")
    self.transformations = transformations

    if device is None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        self.device = device

    # Transformations
    self.toRGB = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x) # grayscale to RGB
    self.resizeBbox = ResizeBbox((640, 640), antialias=False)
    self.colorJitterBbox = ColorJitterBbox(brightness=.5, hue=.01)
    self.horizontalFlipBbox = HorizontalFlipBbox(p=0.5)
    self.randomRotationBbox = RandomRotationBbox(degrees=(-10, 10))
    self.gaussianBlur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.0))

    # load the dataset with the annotations
    self.categories = {}
    self.supercategories = {}
    self.name2category = {}
    self.index2category = {}
    self.loadDataset()

    if size is None:
      self.size = self.dataset.shape[0]
    else:
      self.size = size

  '''
  Get the amount of samples contained in the dataset.
  '''
  def __len__(self):
    return self.size

  '''
  Get a single from a index position.
  '''
  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.dataset.loc[idx, "file_name"])

    image = read_image(img_path).to(self.device)

    bbox = self.dataset.loc[idx, "bbox"]
    bbox_tensor = torch.tensor(bbox, device=self.device)

    image = self.toRGB(image)
    image, bbox_tensor = self.resizeBbox(image, bbox_tensor)

    # Run the transformations
    if self.split == "train" and self.transformations:

      image, bbox_tensor = self.randomRotationBbox(image, bbox_tensor)
      # image, bbox_tensor = self.resizeBbox(image, bbox_tensor)
      image, bbox_tensor = self.colorJitterBbox(image, bbox_tensor)
      image = self.gaussianBlur(image)
      # image, bbox_tensor = self.horizontalFlipBbox(image, bbox_tensor)

    sentence = self.dataset.loc[idx, "sentence"]
    category_id = self.dataset.loc[idx, "category_id"]
    return image, sentence, bbox_tensor, category_id


  '''
  Load the dataset.
  '''
  def loadDataset(self):

    instances_file = os.path.join(self.annotations_dir, 'instances.json')
    refs_file = os.path.join(self.annotations_dir, 'refs(umd).p')

    try:
      instances = json.load(open(instances_file, 'r'))
      f = open(refs_file, 'rb')
    except OSError:
      print(f"Could not open/read the annotations files.")
      sys.exit()

    # parse the annotations, containing the bounding boxes
    annotations = {}
    for ann in instances["annotations"]:
      annotations[ann["id"]] = ann

    # parse the categories
    for i,cat in enumerate(instances['categories']):
      self.categories[cat['id']] = cat['name']
      self.supercategories[cat['id']] = cat['supercategory']
      self.name2category[cat['name']] = cat['id']
      self.index2category[i] = cat['id']

    # parse the pickle file and split it according to the split variable
    df = pd.DataFrame(pickle.load(f))
    if self.split == "test":
      df = df[df["split"] == "test"]
    elif self.split == "val":
      df = df[df["split"] == "val"]
    else:  # fallback to the train set
      df = df[df["split"] == "train"]


    # Fix: remove from the file name the last digit after the last underscore
    df["file_name"] = df["file_name"].apply(lambda file_name: re.sub("_[0-9]+.jpg", ".jpg", file_name))

    df.loc[:,"sentence"] = df["sentences"].apply(lambda sentences: [s["sent"] for s in sentences])
    df.loc[:,"bbox"] = df["ann_id"].apply(lambda ann_id: self.to_xyxy(annotations[ann_id]["bbox"]))
    df = df.explode("sentence").explode("sentence")

    # keep only the needed columns
    self.dataset = df.loc[:,['file_name', 'sentence', 'bbox', 'category_id']]
    self.dataset.reset_index(drop=True, inplace=True)

  '''
  Plot the image stored at a given index in the dataset along with the objects
  inside the image delineated by their bounding boxes and their sentences.

  Args:
    - idx: index of the image to plot
  '''
  def plot_img(self, idx):
    image, sentence, bbox, _ = self.__getitem__(idx)

    # transform the [x,y,w,h] list into a tensor suited for the task
    bbox_tensor = bbox.unsqueeze(0)
    image = draw_bounding_boxes(image, bbox_tensor, width=3, colors=(0,255,0))

    # image.permute is necessary because torch returns a tensor with the
    # channel dimension in the first position. In this way the image is
    # converted by moving that dimension in the last position, i.e.
    # (3,256,256) becomes (256,256,3)
    plt.imshow(image.permute(1,2,0))
    plt.title(sentence)
    plt.tight_layout()
    plt.show()

  '''
  Converts a vector of coordinates in the "xywh" format to "xyxy"

  Args:
      - bbox: a list of 4 coordinates to be converted in the "xyxy" format
  '''
  def to_xyxy(self, bbox):
    bbox_tnsor = torch.tensor(bbox, device=self.device)
    new_bbox = box_convert(bbox_tnsor, 'xywh', 'xyxy')
    return new_bbox.tolist()
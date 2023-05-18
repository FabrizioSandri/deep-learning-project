import os
import pickle
import re
import json
import pandas as pd
import sys

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
from torchvision import transforms
import matplotlib.pyplot as plt 
import numpy as np

class RefCOCOg(Dataset):
  '''
  RefCOCOg dataset parser

  Args:
      - refcocog_path: path to the directory containing the annotations and
        images folders of the refcocog dataset
      - test_set: when set to "test" this class will load the test part of the
          dataset, when "val" it will load the validation set, otherwise it 
          loads the "train" set
      - aggregate_same_image: if set to False each sample of the dataset 
          corresponds to a single object in the image. If set to True, the 
          same boxes for the same image are aggregated together in order to 
          generate a single sample with that image and a list of boxes 
          describing the objects in the image
      - device: either "cpu" or "cuda"
  ''' 
  def __init__(self, refcocog_path, split, aggregate_same_image=False, device=None):
    self.split = split
    self.img_dir = os.path.join(refcocog_path, "images")
    self.annotations_dir = os.path.join(refcocog_path, "annotations")
    self.aggregate_same_image = aggregate_same_image

    if device is None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        self.device = device

    # Transformations used to resize images to 640x640 and to convert
    # grayscale images(if present) to RGB
    self.resize = transforms.Resize((640, 640), antialias=False)
    self.toRGB = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x) # grayscale to RGB
    
    # load the dataset with the annotations
    self.loadDataset()

  '''
  Get the amount of samples contained in the dataset.
  '''
  def __len__(self):
    return self.dataset.shape[0]

  '''
  Get a single from a index position.
  '''
  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.dataset.loc[idx, "file_name"])
    
    image = read_image(img_path).to(self.device)
    image = self.toRGB(image)

    bbox = self.dataset.loc[idx, "bbox"]
    bbox_tensor = bbox_tensor = torch.tensor(bbox, device=self.device)

    # resize the image to 640x640
    orig_width = image.shape[2]
    orig_height = image.shape[1]
    if orig_width != 640 or orig_height != 640:
      image = self.resize(image)
      increment_factor_x = 640/orig_width
      increment_factor_y = 640/orig_height
      
      # resize also the bounding box
      if self.aggregate_same_image:
        bbox_tensor[:,0] *= increment_factor_x
        bbox_tensor[:,1] *= increment_factor_y
        bbox_tensor[:,2] *= increment_factor_x
        bbox_tensor[:,3] *= increment_factor_y
      else:
        bbox_tensor[0] *= increment_factor_x
        bbox_tensor[1] *= increment_factor_y
        bbox_tensor[2] *= increment_factor_x
        bbox_tensor[3] *= increment_factor_y

    # if the dataset should return the same image only once, then return it
    # with a list of bounding boxes
    if self.aggregate_same_image:
      return image, bbox_tensor
    else:
      sentence = self.dataset.loc[idx, "sentence"]
      return image, sentence, bbox_tensor

  '''
  Load the dataset. This function is supposed to combine the same images into
  a single sample if 'aggregate_same_image' is set to True.
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
    
    if self.aggregate_same_image:
      # Group together the same images into a single sample, merging together
      # the bounding boxes and the sentences for each single object
      df.loc[:,"sentence"] = df["sentences"].apply(lambda sentences: [[s["sent"] for s in sentences]])
      df.loc[:,"bbox"] = df["ann_id"].apply(lambda ann_id: [self.to_xyxy(annotations[ann_id]["bbox"])])
      df = df.groupby('image_id').agg({'file_name': 'first', 'sentence': 'sum', 'bbox': 'sum'}).reset_index()
    else:
      df.loc[:,"sentence"] = df["sentences"].apply(lambda sentences: [s["sent"] for s in sentences])
      df.loc[:,"bbox"] = df["ann_id"].apply(lambda ann_id: self.to_xyxy(annotations[ann_id]["bbox"]))
      df = df.explode("sentence").explode("sentence")

    # keep only the needed columns
    self.dataset = df.loc[:,['file_name', 'sentence', 'bbox']]
    self.dataset.reset_index(drop=True, inplace=True)

  '''
  Plot the image stored at a given index in the dataset along with the objects
  inside the image delineated by their bounding boxes and their sentences.

  Args:
    - idx: index of the image to plot
  '''
  def plot_img(self, idx):
    image, sentence, bbox = self.__getitem__(idx)
    
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

import os
import pickle
import re
import json
import pandas as pd

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
import matplotlib.pyplot as plt 
import numpy as np

class RefCOCOg(Dataset):

    '''
    Constructor for the RefCOCOg class

    Args:
        - refcocog_path: path to the directory containing the annotations and
          images folders of the refcocog dataset
        - test_set: when set to True this class will load the test part of the
            dataset
    '''
    def __init__(self, refcocog_path, test_set):
        self.test_set = test_set
        self.img_dir = os.path.join(refcocog_path, "images")
        self.annotations_dir = os.path.join(refcocog_path, "annotations")

        # load the dataset with the annotations
        self.loadDataset()

    '''
    Get the amount of samples contained in the dataset.
    '''
    def __len__(self):
        return self.dataset.shape[0]

    '''
    Get a single item using the index.
    '''
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.dataset.loc[idx, "file_name"])
        
        image = read_image(img_path)
        sentences = self.dataset.loc[idx, "sentences"]
        bbox = self.dataset.loc[idx, "bbox"]
       
        return image, sentences, bbox

    '''
    Load the dataset with the annotations in the form of sentences along with
    the location of the images. This function is supposed to combine the
    sentences of the same image into a single sample.
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

        # parse the annotations, containing the bound boxes
        annotations = {}
        for ann in instances["annotations"]:
            annotations[ann["id"]] = ann
 
        # parse the pickle file and split it according to the test_set variable
        df = pd.DataFrame(pickle.load(f))
        if self.test_set:
            df = df[df["split"] == "test"]
        else:
            df = df[df["split"] != "test"]

        # Fix: remove from the file name the last digit after the last underscore
        df["file_name"] = df["file_name"].apply(lambda file_name: re.sub("_[0-9]+.jpg", ".jpg", file_name))

        # Group together the same images into a single sample, merging together
        # the bounding boxes and the sentences for each single object
        df.loc[:,"sentences"] = df["sentences"].apply(lambda sentences: [[s["sent"] for s in sentences]])
        df.loc[:,"bbox"] = df["ann_id"].apply(lambda ann_id: [annotations[ann_id]["bbox"]])
        df = df.groupby('image_id').agg({'file_name': 'first', 'sentences': 'sum', 'bbox': 'sum'}).reset_index()

        self.dataset = df

    '''
    Plot the image stored at a given index in the dataset along with the objects
    inside the image delineated by their bounding boxes and their sentences.

    Args:
        - idx: index of the image to plot
    '''
    def plot_img(self, idx):
        image, sentences, bbox = self.__getitem__(idx)
        
        num_objs = len(bbox)
        for obj in range(num_objs):
            print(f"================ Object %d ================" % obj)
            for sentence in sentences[obj]:
                print("- %s" % sentence)

            # transform the [x,y,w,h] list into a tensor suited for the task
            bbox_tensor = torch.tensor(bbox[obj])
            bbox_tensor = box_convert(bbox_tensor, 'xywh', 'xyxy')
            bbox_tensor = bbox_tensor.unsqueeze(0)
            image = draw_bounding_boxes(image, bbox_tensor, width=3, colors=(0,255,0))
            plt.text(bbox_tensor[:,0], bbox_tensor[:,1], f"Object %d" % obj, fontsize=12, bbox=dict(facecolor='green'))

        # image.permute is necessary because torch returns a tensor with the
        # channel dimension in the first position. In this way the image is
        # converted by moving that dimension in the last position, i.e.
        # (3,256,256) becomes (256,256,3)
        plt.imshow(image.permute(1,2,0))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    train_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", test_set=False)
    test_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", test_set=True)

    random_idx = np.random.randint(0, len(train_dataset))
    train_dataset.plot_img(random_idx)
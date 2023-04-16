import os
import pickle
import re
import pandas as pd
import numpy as np

from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 


class RefCOCOg(Dataset):

    def __init__(self, annotations_file, img_dir):
        self.annotations_file = annotations_file
        self.img_dir = img_dir

        # load the annotations
        self.loadAnnotations()

    def loadAnnotations(self): 
        try:
            f = open(self.annotations_file, 'rb')
        except OSError:
            print(f"Could not open/read file: " % self.annotations_file)
            sys.exit()

        # parse the pickle file
        df = pd.DataFrame(pickle.load(f))

        # Fix: remove from the file name the last digit after the last underscore
        df["file_name"] = df["file_name"].apply(lambda file_name: re.sub("_[0-9]+.jpg", ".jpg", file_name))

        self.annotations = df

    def __len__(self):
        return len(self.annotations.shape[0])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.loc[idx, "file_name"])
        
        image = read_image(img_path)
        sentences = [ sentence["sent"] for sentence in self.annotations.loc[idx, "sentences"] ]
       
        return image, sentences

    def plot_img(self, idx):
        image, sentences = self.__getitem__(idx)

        plt.title("\n".join(sentences))
        plt.imshow(image.permute(1,2,0))
       
        plt.show()

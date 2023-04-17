import os
import pickle
import re
import pandas as pd

from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
import numpy as np

class RefCOCOg(Dataset):

    '''
    Constructor for the RefCOCOg class

    Args:
        - annotations_file: path to the file containing the annotations
        - img_dir: path to the directory containing the images
        - test_set: when set to True this class will load the test part of the
            dataset
    '''
    def __init__(self, annotations_file, img_dir, test_set):
        self.img_dir = img_dir
        self.test_set = test_set

        # load the dataset with the annotations
        self.loadDataset(annotations_file)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.dataset.loc[idx, "file_name"])
        
        image = read_image(img_path)
        sentences = self.dataset.loc[idx, "sentences"]
       
        return image, sentences

    '''
    Load the dataset with the annotations in the form of sentences along with
    the location of the images. This function is supposed to combine the
    sentences of the same image into a single sample.

    Args:
        - annotations_file: path to the file containing the annotations
    '''
    def loadDataset(self, annotations_file): 
        try:
            f = open(annotations_file, 'rb')
        except OSError:
            print(f"Could not open/read file: " % annotations_file)
            sys.exit()

        # parse the pickle file
        df = pd.DataFrame(pickle.load(f))

        # Fix: remove from the file name the last digit after the last underscore
        df["file_name"] = df["file_name"].apply(lambda file_name: re.sub("_[0-9]+.jpg", ".jpg", file_name))

        if self.test_set:
            df = df[df["split"] == "test"]
        else:
            df = df[df["split"] == "train"]

        # group together the same images into a single sample
        df = df.groupby('image_id').agg({'file_name': 'first', 'sentences': 'sum'}).reset_index()
        df["sentences"] = df["sentences"].apply(lambda sentences: [s["sent"] for s in sentences])
        self.dataset = df


    '''
    Plot an image along with the sentences associated to it.

    Args:
        - idx: index of the image to plot
    '''
    def plot_img(self, idx):
        image, sentences = self.__getitem__(idx)

        plt.title("\n".join(sentences))
        plt.imshow(image.permute(1,2,0))
        plt.tight_layout()
        
        plt.show()


if __name__ == '__main__':

    train_dataset = RefCOCOg("/home/fabri/Downloads/refcocog/annotations/refs(umd).p", "/home/fabri/Downloads/refcocog/images", test_set=False)
    test_dataset = RefCOCOg("/home/fabri/Downloads/refcocog/annotations/refs(umd).p", "/home/fabri/Downloads/refcocog/images", test_set=True)

    random_idx = np.random.randint(0, len(train_dataset))
    train_dataset.plot_img(random_idx)
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
from tqdm import tqdm

from dataset.dataset import RefCOCOg
from models.train_model import TrainRPN
from models.model import RPN
from utilities import Utilities 

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

################################################################################

# Dataloaders with a sample for each object
train_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", split="train")
val_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", split="val")
test_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", split="test")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=20)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=20)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=20)


# This custom collate function is designed for the dataloader to load data
# grouped by image. It returns a batch of stacked images and a list of bounding
# boxes for each element in the batch.
def grouped_collate_batch(batch):
  images = []
  boxes = []
  
  for (img,bbox) in batch:
    images.append(img)
    boxes.append(bbox)

  return torch.stack(images), boxes

# Dataloaders grouping samples by image
grouped_train_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", split="train", aggregate_same_image=True)
grouped_val_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", split="val", aggregate_same_image=True)

grouped_train_loader = DataLoader(grouped_train_dataset, shuffle=True, batch_size=10, collate_fn=grouped_collate_batch) 
grouped_val_loader = DataLoader(grouped_val_dataset, shuffle=False, batch_size=10, collate_fn=grouped_collate_batch) 

############################## TRAINING ##############################
model_trainer = TrainRPN(device=device)
model, _, _  = model_trainer.train(num_epochs=2, train_loader=grouped_train_loader, val_loader=grouped_val_loader)

torch.save(model.state_dict(), "model.pt")

############################## EVALUATION ##############################

model = RPN(device=device)
model.load_state_dict(torch.load('model.pt'))
model.train(False)

def inference(image_id, count):
  image, _, _ = test_dataset[image_id]
  pred_bbox, _ = model.forward(image.unsqueeze(0), ground_truth_boxes=None)

  modified_img = draw_bounding_boxes(image, pred_bbox[0][:count], width=2, colors=(0,255,0))

  plt.imshow(modified_img.cpu().permute(1,2,0))
  plt.show()

inference(image_id=1, count=50)
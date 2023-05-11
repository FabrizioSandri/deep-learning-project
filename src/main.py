import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
from tqdm import tqdm

from dataset import RefCOCOg
from baseline import YoloBaseline 
from utilities import Utilities 

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

################################################################################

train_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", test_set=False)
test_dataset = RefCOCOg("/home/fabri/Downloads/refcocog", test_set=True)

train, val = random_split(train_dataset, [0.8, 0.2])

train_loader = DataLoader(train, shuffle=True, batch_size=20) 
val_loader = DataLoader(val, shuffle=True, batch_size=20)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=20)

############################## BASELINE ##############################
image, sentence, bbox = train_dataset[4]

# Yolo Baseline
yolobaseline = YoloBaseline()
predicted = yolobaseline.inference(image, "pizza", plot=False)

# transform the [x,y,w,h] list into a tensor suited for the task
bbox_tensor = torch.tensor(predicted, device=device).unsqueeze(0)
image = draw_bounding_boxes(image, bbox_tensor, width=3, colors=(0,255,0))

plt.imshow(image.permute(1,2,0))
plt.tight_layout()
plt.show()

############################## MEASUREMENTS ##############################
yolobaseline = YoloBaseline()
utilities = Utilities()

overall_intersection = 0
overall_union = 0

total_iou = 0

# data loader for the test_set without batches
test_loader_no_batch = DataLoader(test_dataset, shuffle=False, batch_size=None)

for image, description, true_bbox in tqdm(test_loader_no_batch):
  
  # inference with the baseline
  pred_bbox = yolobaseline.inference(image, description)

  intersection = utilities.bbox_intersection(true_bbox, pred_bbox)
  union = utilities.bbox_union(true_bbox, pred_bbox)
  
  overall_intersection += intersection
  overall_union += union
  
  total_iou += intersection/union

print("")
print("oIOU : %f" %(overall_intersection/overall_union))
print("mIOU : %f" % (total_iou/len(test_loader_no_batch)))
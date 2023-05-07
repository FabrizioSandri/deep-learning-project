import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
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

train_loader = DataLoader(train_dataset, shuffle=False, num_workers=4, batch_size=None, pin_memory=True ) # TODO: remember to remove batch_size=1 
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=None, pin_memory=True )

############################## BASELINE ##############################
image, descriptions, bboxes = train_dataset[4]

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
num_examples = 0

# for sample_i in tqdm(range(len(train_dataset))):
for image, descriptions, bboxes in tqdm(test_loader):
  
  # variables used to compute the mean intersection over union(this is computed on each single example)
  mean_intersection = 0
  mean_union = 0
  num_examples += 1

  for obj_i in range(len(bboxes)):  # iterate over the objects
    true_bbox = torch.tensor(bboxes[obj_i], device=device) # get the real bounding box of that object
    
    for txt in descriptions[obj_i]: # iterate over the descriptions of the object
      
      # inference with the baseline
      pred_bbox = yolobaseline.inference(image, txt)

      # print(txt)
      # result_img = draw_bounding_boxes(image, true_bbox, width=3, colors=(255,0,0))
      # result_img = draw_bounding_boxes(result_img, pred_bbox, width=3, colors=(0,255,0))
      # plt.imshow(result_img.permute(1,2,0))
      # plt.show()

      intersection = utilities.bbox_intersection(true_bbox, pred_bbox)
      union = utilities.bbox_union(true_bbox, pred_bbox)
      overall_intersection += intersection
      overall_union += union

      mean_intersection += intersection
      mean_union += union
  
  total_iou += mean_intersection/mean_union

print("")
print("oIOU : %f" %(overall_intersection/overall_union))
print("mIOU : %f" % (total_iou/num_examples))
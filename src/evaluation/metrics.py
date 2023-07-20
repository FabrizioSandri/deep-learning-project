import torch
import torchvision 
import torchmetrics
from torchmetrics.classification import MulticlassRecall

device = "cuda" if torch.cuda.is_available() else "cpu"

def IOU(gtBoxes, predictedBoxes):
  iouMatrix = torchvision.ops.box_iou(gtBoxes, predictedBoxes).diag()
  return iouMatrix

def recall(predClasses, gtClasses):
  num_categories = 91
  recall = MulticlassRecall(num_classes=num_categories, average='none').to(device)
  return (recall(predClasses, gtClasses)[predClasses])

def cosineSimilarity(encoded_crops, encoded_descriptions):
  return torch.nn.functional.cosine_similarity(encoded_crops, encoded_descriptions)
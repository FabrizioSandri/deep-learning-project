from torchvision import transforms
import torch
import torchvision
import clip
from tqdm import tqdm
from evaluation.metrics import *

# This function is utilized to retrieve the cropped image from the provided
# image, taking into account the specified bounding box. Handle the case of a
# zero size bounding box by returning the original image
def cropBboxImage(image, bbox):
  try:
    return transforms.functional.crop(image, int(bbox[1].item()), int(bbox[0].item()), int(bbox[3].item()), int(bbox[2].item()))
  except:
    return image

def compute_metrics(model, test_loader):
  total_cosine_sim = 0
  total_recall = 0
  total_iou = 0
  num_samples = len(test_loader.dataset)

  clipModel, preprocess = clip.load("RN50", model.device)
  clipModel = clipModel.float()
  clipModel.requires_grad = False  # Freeze clip model

  # Encode the dataset categories using CLIP
  encoded_categories = model.encode_text(test_loader.dataset.categories.values())

  # sets the model in evaluation mode and disable gradient tracking
  model.eval()
  with torch.no_grad():
    for images, descriptions, gtBoxes, gtClasses in tqdm(test_loader):
        gtClasses = gtClasses.to(model.device)
        batch_size = images.shape[0]

        # The predicted bounding box is the best proposal accordint to the score
        proposals, _ = model(images, descriptions, gtBoxes=None)
        best_proposals = torch.zeros((batch_size, 4), device=model.device)
        for j in range(batch_size):
            best_proposals[j,:] = proposals[j][0]

        # Crop the predictions from the original image and encode them using the
        # image encoder of clip
        resized = torch.zeros((batch_size,3,224,224), device=model.device)
        for i in range(batch_size):
            cropped_image = cropBboxImage(images[i], best_proposals[i])
            resized[i,:,:,:] = preprocess(model.toPIL(cropped_image))

        encoded_crops = clipModel.encode_image(resized)
        encoded_crops /= encoded_crops.norm(dim=-1, keepdim=True)

        # Encode the descriptions
        encoded_descriptions = model.encode_text(descriptions)

        # Calculate the cosine similarity between each encoded image and each encoded
        # category
        similarity_matrix = torch.matmul(encoded_crops, encoded_categories.T)
        predClasses = torch.argmax(similarity_matrix, dim=1)
        for i in range(batch_size): # convert back the indices to category indices
          predClasses[i] = test_loader.dataset.index2category[int(predClasses[i].item())]

        # compute the metrics
        total_iou += IOU(gtBoxes, best_proposals).sum()
        total_recall += recall(predClasses, gtClasses).sum()
        total_cosine_sim += cosineSimilarity(encoded_crops, encoded_descriptions).sum()

  print(f"IOU: {total_iou / num_samples}")
  print(f"Recall: {total_recall / num_samples}")
  print(f"Cosine similarity: {total_cosine_sim / num_samples}")

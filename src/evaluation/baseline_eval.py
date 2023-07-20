from torchvision import transforms
import torch
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

# Encode a batch of textual descriptions using CLIP
def encode_text(clipModel, texts):
  tokenized_description = clip.tokenize(texts).cuda()
  text_features = clipModel.encode_text(tokenized_description).float()
  text_features /= text_features.norm(dim=-1, keepdim=True)
  return text_features

def compute_baseline_metrics(model, test_loader):

  total_cosine_sim = 0
  total_recall = 0
  total_iou = 0
  num_samples = len(test_loader.dataset)

  toPIL = transforms.ToPILImage()

  # sets the model in evaluation mode and disable gradient tracking
  model.eval()
  with torch.no_grad():
    for images, descriptions, gtBoxes, gtClasses in tqdm(test_loader):
        gtClasses = gtClasses.to(model.device)
        batch_size = images.shape[0]

        best_proposals = torch.zeros((batch_size, 4), device=model.device)
        pred_classes = torch.zeros((batch_size,), dtype=torch.long, device=model.device)
        for j in range(batch_size):
          # The predicted bounding box is the best proposal according to the score
          proposal, pred_class = model.inference(images[j], descriptions[j])
          best_proposals[j, :] = proposal
          pred_classes[j] = int(pred_class)

        # Crop the predictions from the original image and encode them using the
        # image encoder of clip
        resized = torch.zeros((batch_size,3,224,224), device=model.device)
        for i in range(batch_size):
            cropped_image = cropBboxImage(images[i], best_proposals[i])
            resized[i,:,:,:] = model.preprocess(toPIL(cropped_image))

        encoded_crops = model.clipModel.encode_image(resized)
        encoded_crops /= encoded_crops.norm(dim=-1, keepdim=True)

        # Encode the descriptions
        encoded_descriptions = encode_text(model.clipModel, descriptions)

        # compute the metrics
        total_iou += IOU(gtBoxes, best_proposals).sum()
        total_recall += recall(pred_classes, gtClasses).sum()
        total_cosine_sim += cosineSimilarity(encoded_crops, encoded_descriptions).sum()

  print(f"IOU: {total_iou / num_samples}")
  print(f"Recall: {total_recall / num_samples}")
  print(f"Cosine similarity: {total_cosine_sim / num_samples}")

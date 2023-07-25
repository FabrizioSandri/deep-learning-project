import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

def plotPrediction(model, image, text, gtBox=None):
    model.eval()
    with torch.no_grad():
      proposals, _ = model(image.unsqueeze(0), [text], gtBoxes=None)  # inference
      
      best_image = image
      if gtBox is not None:
        best_image = draw_bounding_boxes(best_image, gtBox.unsqueeze(0), width=5, colors=(0,255,0)) # ground truth
      best_image = draw_bounding_boxes(best_image, proposals[0][0].unsqueeze(0), width=5, colors=(255,0,0))  # prediction

      plt.imshow(best_image.permute(1,2,0))
      plt.title(text, wrap=True)
      plt.show()
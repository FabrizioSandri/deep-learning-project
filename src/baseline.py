import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip

from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
from torchvision import transforms

class YoloBaseline():

  def __init__(self, device=None):
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    current_backend = matplotlib.get_backend()

    # models definition
    self.yoloModel = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False, pretrained=True).to(self.device)
    self.clipModel, self.preprocess = clip.load("RN50", self.device)

    matplotlib.use(current_backend)

  '''
  Runs inference and returns a bounding box in the format 'xyxy'. Note that if 
  no matching bounding box is found by this method, the function returns None.

  Args:
      - image: a torch tensor defining the image
      - description: a textual description 
      - plot: if set to true plots all the bounding boxes with their similarity
  '''
  def inference(self, image, description, plot=False):

    #################### YOLO ####################
    toPIL = transforms.ToPILImage()
    image = toPIL(image)

    # Inference
    yolo_results = self.yoloModel(image).pandas()
    cropped_images = self.crop_bbox(yolo_results, image)

    # Stop here if YOLO doesn't find any bounding box
    if len(cropped_images) == 0:
      return None

    #################### CLIP ####################
    preprocessed_images = self.preprocess_images(cropped_images)
    tokenized_description = clip.tokenize([description]).cuda()

    # latent representations of the cropped images and the textual description
    with torch.no_grad():
      image_features = self.clipModel.encode_image(preprocessed_images)
      text_features = self.clipModel.encode_text(tokenized_description).float()
    
    # compute the cosine similarity between each pair of images and the textual 
    # descriptions
    similarities = []
    for obj_i in range(image_features.shape[0]):
      sim = self.cosine_similarity(image_features[obj_i,:], text_features)
      similarities.append(sim)

    if plot:
      self.plot_results(similarities, cropped_images)

    best_bbox = np.argmax(similarities)

    # get the best matching bounding box returned by yolo
    res = yolo_results.xyxy[0]
    xmin = res.loc[best_bbox,"xmin"]
    ymin = res.loc[best_bbox,"ymin"]
    xmax = res.loc[best_bbox,"xmax"]
    ymax = res.loc[best_bbox,"ymax"]
    pred_bbox = torch.tensor([xmin, ymin, xmax, ymax], device=self.device)

    return pred_bbox

  '''
  This method plots all the bounding boxes found by this Baseline with the
  corresponding value of similarity 
  '''
  def plot_results(self, similarities, cropped_images):
    nrows = int(np.ceil(len(similarities)/4))
    ncols = 4
    fig, ax = plt.subplots(nrows, ncols, subplot_kw=dict(box_aspect=1), figsize=(10, 10))
    ax = ax.flatten()
    for img_i in range(len(similarities)):
      ax[img_i].set_title(round(similarities[img_i], 4))
      ax[img_i].imshow(cropped_images[img_i], aspect='auto')
      ax[img_i].axis('off')

    fig.suptitle("Bounding boxes similarities with the given text")
    plt.tight_layout()
    plt.show()

  '''
  Given as input the results returned by yolo by running inference on 'image',
  this method returns a list of images as a result of cropping them according
  to the bounding boxes detected by YOLO

  Args:
    - yolo_results: pandas dataframe returned by Yolo after running inference
    - image: the image fed as input to Yolo
  '''
  def crop_bbox(self, yolo_results, image):
    num_boxes = len(yolo_results.xyxy[0])
    cropped_images = []
    
    res = yolo_results.xyxy[0]
    for i in range(num_boxes):
      xmin = res.loc[i,"xmin"]
      ymin = res.loc[i,"ymin"]
      xmax = res.loc[i,"xmax"]
      ymax = res.loc[i,"ymax"]
      bbox = torch.tensor([xmin, ymin, xmax, ymax], device=self.device)
      bbox = box_convert(bbox, 'xyxy', 'xywh')            

      cropped = transforms.functional.crop(image, int(bbox[1].item()), int(bbox[0].item()), int(bbox[3].item()), int(bbox[2].item()))
      cropped_images.append(cropped)

    return cropped_images

  '''
  Preprocess a list of cropped images, by stacking all of them into a single 
  tensor. For example if this method takes as input 5 images of different sizes,
  then the result is a tensor of size [5,3,224,224] i.e. the first dimensions is
  for the number of images, the second one for the number of color planes(3 in 
  RGB) and the last twos describe the width and height of each image.

  Args:
    - cropped_images: a list of cropped images
  '''
  def preprocess_images(self, cropped_images):

    # preprocess with CLIP each cropped PIL image(converts each image in a image
    # of size [3,224,224])
    preprocessed = []
    for image in cropped_images:
      processed_img = self.preprocess(image).to(self.device)
      preprocessed.append(processed_img)

    return torch.stack(preprocessed).to(self.device) # return a single tensor


  '''
  This method computes the cosine similarity between the latent representation 
  of the text and the latent representation of an image. 
  '''
  def cosine_similarity(self, image_encoding, text_encoding):
    cos = torch.nn.CosineSimilarity()
    return float(cos(image_encoding, text_encoding).item())
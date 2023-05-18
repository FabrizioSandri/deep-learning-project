import torch
import clip

from torchvision import transforms
import torchvision.models.detection.rpn as rpn
import torchvision.models.detection.image_list as il
from torchvision.models.detection.anchor_utils import AnchorGenerator

class RPN(torch.nn.Module):

  def __init__(self, device=None):
    super(RPN, self).__init__()
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    self.clipModel, self.preprocess = clip.load("RN50", self.device)
    self.clipModel = self.clipModel.float()   # self.clipModel.float() is necessary otherwise clip outputs NAN 
    self.clipModel.requires_grad = False  # Freeze clip model
    self.clipModel.visual.attnpool = torch.nn.Identity()  # remove the last attention layer of the CLIP ResNet

    self.toPIL = transforms.ToPILImage()

    # predefined anchors
    self.rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                aspect_ratios=((0.5, 1.0, 2.0),)).to(self.device)

    feature_map_channels = 2048
    self.rpn_head = rpn.RPNHead(
        feature_map_channels, self.rpn_anchor_generator.num_anchors_per_location()[0]
    ).to(self.device)

    rpn_pre_nms_top_n = {"training": 2000, "testing": 1000}
    rpn_post_nms_top_n = {"training": 2000, "testing": 1000}
    rpn_nms_thresh = 0.7
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.2
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5

    self.rpn = rpn.RegionProposalNetwork(
        self.rpn_anchor_generator,
        self.rpn_head,
        rpn_fg_iou_thresh, 
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image, 
        rpn_positive_fraction,
        rpn_pre_nms_top_n, 
        rpn_post_nms_top_n, 
        rpn_nms_thresh).to(self.device)

  def forward(self, images, ground_truth_boxes):

    batch_size = images.shape[0]

    with torch.no_grad():

      # preprocess the images 
      resized = torch.zeros((batch_size,3,224,224), device=self.device)
      for i in range(batch_size):
        resized[i,:,:,:] = self.preprocess(self.toPIL(images[i,:,:,:]))
    
      feature_map = self.clipModel.visual(resized)

    feature_maps = {"0": feature_map} # pass only one feature map of size 2048 x 7 x 7
    image_sizes = [(640,640)] * batch_size
    image_list = il.ImageList(images, image_sizes)

    return self.rpn(image_list, feature_maps, ground_truth_boxes)


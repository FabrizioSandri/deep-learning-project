from torchvision import transforms
import torchvision.models.detection.rpn as rpn
import torch
import torchvision
import clip

from rpn.AnchorGenerator import *
from rpn.RPNHead import *
from fusion_modules.FusionModule_Concatenation import *
from fusion_modules.FusionModule_CrossAttention import *

# Wrapper class used to pass information regarding the size of the images given
# as input
class ImageList:
    def __init__(self, image_sizes):
        self.image_sizes = image_sizes

class ClipRPN(torch.nn.Module):

  def __init__(self, anchors_scales, anchor_ratios, fusion_type="cross_attention", attention_heads=8, feature_map_channels=2048, hidden_dim=1024, dropout_rate=0.1, device=None):
    super().__init__()

    if device is None:
      self.device = "cuda" if torch.cuda.is_available () else "cpu"
    else:
      self.device = device

    # Load CLIP ResNet50 and remove it's last attention layer to expose it's
    # backbone
    self.clipModel, self.preprocess = clip.load("RN50", self.device)
    self.clipModel = self.clipModel.float()
    self.clipModel.requires_grad = False  # Freeze clip model
    self.clipModel.visual.attnpool = torch.nn.Identity()  # remove the last attention layer of the CLIP ResNet

    self.toPIL = transforms.ToPILImage()

    # Prepare the Region proposal network with it's anchors and the RPN head
    self.anchor_generator = AnchorGenerator(anchors_scales, anchor_ratios)
    rpn_head = RPNHead(feature_map_channels, len(self.anchor_generator))

    rpn_pre_post_nms_top_n = {"training": 200, "testing": 100}
    self.rpn_wrapper = rpn.RegionProposalNetwork(
        self.anchor_generator, rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.2,          # anchor boxes with a IOU < 0.2 are considered negative
        batch_size_per_image=256,   # for each image 256 anchors are sampled
        positive_fraction=0.5,      # out of the 256 anchors half are positive and half negative
        pre_nms_top_n=rpn_pre_post_nms_top_n, post_nms_top_n=rpn_pre_post_nms_top_n,
        nms_thresh=0.7              # threshold for Non maximum suppression
    )

    # Fusion module
    if fusion_type == "attention":
      self.fusion_module = FusionModule_CrossAttention(
          feature_map_channels=feature_map_channels,
          nhead=attention_heads,      # number of parallel attention heads
          dim_feedforward=hidden_dim, # hidden units in the feed forward block
          kv_dim=1024,                # embedding size of key and values(1024 corresponding ot CLIP's text embedding size)
          dropout_rate=dropout_rate
      )
    else:
      self.fusion_module = FusionModule_Concatenation(
          feature_map_channels=feature_map_channels,
          out_feature_map_channels=feature_map_channels,
          text_embedding_size=1024,
          dropout_rate=dropout_rate
      )

  # Encode a batch of textual descriptions using CLIP
  def encode_text(self, texts):
    tokenized_description = clip.tokenize(texts).cuda()
    text_features = self.clipModel.encode_text(tokenized_description).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

  # Obtain the feature maps from a batch of images by passing them through the
  # ResNet50 of CLIP
  def getFeatureMaps(self, images):
    batch_size = images.size(0)

    resized = torch.zeros((batch_size,3,224,224), device=self.device)
    for i in range(batch_size):
      resized[i,:,:,:] = self.preprocess(self.toPIL(images[i,:,:,:]))

    feature_maps = self.clipModel.visual(resized)
    return feature_maps

  # By setting gtBoxes=None the model runs in inference mode
  def forward(self, images, texts, gtBoxes=None):
    batch_size = images.size(0)

    # Get the textual embeddings and the feature map
    text_embedding = self.encode_text(texts)                    # [N, 1024]
    feature_maps = self.getFeatureMaps(images)                  # [N, 2048, 7, 7]

    # Feature map fusion using the fusion module
    fused_fm = self.fusion_module(feature_maps, text_embedding)  # [N, 2048, 7, 7]

    ground_truth_boxes = None
    if gtBoxes is not None:
      ground_truth_boxes = [{"boxes": gtBoxes[batch_i].unsqueeze(0)} for batch_i in range(batch_size)]

    image_sizes = ImageList([(640,640)] * batch_size)
    return self.rpn_wrapper(image_sizes, {"0": fused_fm}, ground_truth_boxes)

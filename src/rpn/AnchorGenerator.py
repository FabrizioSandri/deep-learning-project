import torch
import math

from torchvision.ops import box_convert


class AnchorGenerator(torch.nn.Module):

  def __init__(self, scales, ratios, num_centers=7, spatial_dim=640, device=None):
    super().__init__()

    if device is None:
      self.device = "cuda" if torch.cuda.is_available () else "cpu"
    else:
      self.device = device

    anchorCenters = self.getAnchorCenters(num_centers, spatial_dim)
    self.anchors = self.getAnchorBoxes(anchorCenters, scales, ratios, spatial_dim)
    self.anchors_per_pixel = len(scales) * len(ratios)

  def forward(self, images, *arg):
    batch_size = len(images.image_sizes)
    return [self.anchors] * batch_size

  def __len__(self):
    return self.anchors_per_pixel

  '''
  Returns the anchor centers evenly spaced along the spatial dimension
  specified. The parameter 'num_centers' should correspond with the spatial
  dimension of the feature map.

  Returns tensor [A]
  '''
  def getAnchorCenters(self, num_centers, spatial_dim):
      interval = math.floor(spatial_dim / num_centers)
      return torch.arange(0, spatial_dim - interval, interval)

  '''
  Starting from the given 'anchorCenters' this function generates all the
  possible anchor boxes that can be formed combining 'scales' and 'ratios'.

  Returns tensor [N, K]
  '''
  def getAnchorBoxes(self, anchorCenters, scales, ratios, spatial_dim):

      num_centers = anchorCenters.size(0)
      num_scales = scales.size(0)
      num_ratios = ratios.size(0)
      num_anchors_per_pixel = num_scales * num_ratios

      # compute the combinations of widths and heights according to the rations
      # and scales
      wh_combinations = torch.zeros((num_scales * num_ratios, 2), device=self.device)

      h_ratios = torch.sqrt(ratios)
      w_ratios = 1 / h_ratios

      i = 0
      for ratio_i in range(ratios.size(0)):
        for scale in scales:
          wh_combinations[i,0] = scale * w_ratios[ratio_i] # width
          wh_combinations[i,1] = scale * h_ratios[ratio_i] # height
          i += 1

      wh_combinations = wh_combinations.repeat(num_centers * num_centers,1)

      # compute the combinations of centers positions
      centers = torch.zeros((num_centers * num_centers, 2), device=self.device)
      i = 0
      for cy in anchorCenters:
        for cx in anchorCenters:
          centers[i,0] = cx
          centers[i,1] = cy
          i += 1

      anchors = centers.repeat_interleave(repeats=num_anchors_per_pixel, dim=0)
      anchors = torch.cat([anchors, wh_combinations], dim=1)
      anchors_xyxy = box_convert(anchors, 'cxcywh', 'xyxy')

      return anchors_xyxy.round()
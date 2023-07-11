import torch
import torchvision

from torchvision import transforms
from torchvision.ops import box_convert

class ColorJitterBbox(transforms.ColorJitter):
  def forward(self, img, bbox):
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
        self.brightness, self.contrast, self.saturation, self.hue
    )

    for fn_id in fn_idx:
        if fn_id == 0 and brightness_factor is not None:
            img = transforms.functional.adjust_brightness(img, brightness_factor)
        elif fn_id == 1 and contrast_factor is not None:
            img = transforms.functional.adjust_contrast(img, contrast_factor)
        elif fn_id == 2 and saturation_factor is not None:
            img = transforms.functional.adjust_saturation(img, saturation_factor)
        elif fn_id == 3 and hue_factor is not None:
            img = transforms.functional.adjust_hue(img, hue_factor)

    return img, bbox

class RandomRotationBbox(transforms.RandomRotation):
  def forward(self, img, bbox):
    fill = self.fill
    channels, _, _ = transforms.functional.get_dimensions(img)
    if isinstance(img, torch.Tensor):
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * channels
        else:
            fill = [float(f) for f in fill]
    angle = self.get_params(self.degrees)

    cxcywh_format = box_convert(bbox, 'xyxy', 'cxcywh')
    center = (cxcywh_format[0], cxcywh_format[1])
    return transforms.functional.rotate(img, angle, self.interpolation, self.expand, center, fill), bbox

class HorizontalFlipBbox(transforms.RandomHorizontalFlip):
  def forward(self, image, bbox) :
    if torch.rand(1) < self.p:
      image = transforms.functional.hflip(image)
      cxcywh_format = box_convert(bbox, 'xyxy', 'cxcywh')

      # flip the y coordinate
      cxcywh_format[0] = image.shape[2] - cxcywh_format[0]

      bbox = box_convert(cxcywh_format, 'cxcywh', 'xyxy')

    return image, bbox

# Auxiliary augmentation function used to resize an image including it's
# associated bounding box
class ResizeBbox(transforms.Resize):
  def forward(self, image, bbox) :
    orig_width = image.shape[2]
    orig_height = image.shape[1]

    # resize only if needed
    if orig_width != self.size[1] or orig_height != self.size[0]:
      image = transforms.functional.resize(image, self.size, self.interpolation, self.max_size, self.antialias)
      increment_factor_x = self.size[1]/orig_width
      increment_factor_y = self.size[0]/orig_height

      # resize also the bounding box
      bbox[0] *= increment_factor_x
      bbox[1] *= increment_factor_y
      bbox[2] *= increment_factor_x
      bbox[3] *= increment_factor_y

    return image, bbox
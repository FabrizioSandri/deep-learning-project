import torch

class Utilities():

  def __init__(self, device=None):
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

  '''
  Computes the union of two bounding boxes in the "xyxy" format

  Args
    - bbox1 first bounding box in the format(xyxy)
    - bbox2: second bounding box in the format(xyxy)
  '''
  def bbox_union(self, bbox1, bbox2):

      if bbox1 is None:
        area1 = 0.0
      else: 
        bbox1 = torch.tensor(bbox1, device=self.device)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

      if bbox2 is None:
        area2 = 0.0
      else:
        bbox2 = torch.tensor(bbox2, device=self.device)
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

      intersection = self.bbox_intersection(bbox1, bbox2)
      union = area1 + area2 - intersection
      
      return float((union).item())

  '''
  Computes the intersection of two bounding boxes in the "xyxy" format

  Args
    - bbox1 first bounding box in the format(xyxy)
    - bbox2: second bounding box in the format(xyxy)
  '''
  def bbox_intersection(self, bbox1, bbox2):
    
    # if one bbox is not given then the intersection is 0
    if bbox1 is None or bbox2 is None:
      return 0.0

    # coordinate format conversion
    bbox1 = torch.tensor(bbox1, device=self.device)
    bbox2 = torch.tensor(bbox2, device=self.device)

    # compute the coordinates of the inner intersection box
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # compute the intersection
    intersection =  (x_right - x_left) * (y_bottom - y_top)

    return float((intersection).item())

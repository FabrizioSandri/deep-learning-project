import torch
import torchvision

class RPNHead(torch.nn.Module):

    def __init__(self, feature_map_channels, anchors_per_pixel):
        super().__init__()

        # The convolutions used to calculate the objectness and the regression offsets
        self.conv = torchvision.ops.Conv2dNormActivation(feature_map_channels, feature_map_channels, kernel_size=3, activation_layer=torch.nn.ReLU, norm_layer=None)
        self.cls_logits = torch.nn.Conv2d(feature_map_channels, anchors_per_pixel, kernel_size=1, stride=1)
        self.reg_offsets = torch.nn.Conv2d(feature_map_channels, anchors_per_pixel * 4, kernel_size=1, stride=1)

        # Convolutions initialization taken from the original pytorch implementation
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, feature_maps):
        feature_map = feature_maps[0]

        activated_fm = self.conv(feature_map)
        cls_logits = self.cls_logits(activated_fm)
        bbox_reg = self.reg_offsets(activated_fm)

        return [cls_logits], [bbox_reg]
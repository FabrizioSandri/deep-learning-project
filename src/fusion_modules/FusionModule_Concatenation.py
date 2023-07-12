import torch

class FusionModule_Concatenation(torch.nn.Module):

    def __init__(self, feature_map_channels=2048, out_feature_map_channels=2048, text_embedding_size=1024,
                dropout_rate=0.1):
        super().__init__()

        self.conv1x1 = torch.nn.Conv2d(feature_map_channels + 2*text_embedding_size, out_feature_map_channels, kernel_size=(1,1))

    def forward(self, feature_map, text_embedding):
        repeated_text = text_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1,2,7,7)
        concatenated_feature_map = torch.cat([feature_map, repeated_text], dim=1)

        # Final convolution block
        return self.conv1x1(concatenated_feature_map)
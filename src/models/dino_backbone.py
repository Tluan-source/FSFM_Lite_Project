import torch
import torch.nn as nn

class DINOBackbone(nn.Module):
   def __init__(self, model_name="dinov2_vitb14"):
      super().__init__()
      self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

   def forward(self, x):
      features = self.backbone.forward_features(x)
      cls_token = features["x_norm_clstoken"]
      patch_tokens = features["x_norm_patchtokens"]
      return cls_token, patch_tokens
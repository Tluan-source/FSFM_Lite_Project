import torch
import torch.nn as nn
from .dino_backbone import DINOBackbone
from .threec_module import ThreeCModule
from .classifier_head import ClassifierHead

class FSFMLite(nn.Module):
   def __init__(self):
      super().__init__()
      self.backbone = DINOBackbone()
      self.threec = ThreeCModule()
      self.head = ClassifierHead()

   def forward(self, x):
      cls_token, patch_tokens = self.backbone(x)
      enhanced_tokens = self.threec(patch_tokens,cls_token)
      pooled = enhanced_tokens.mean(dim=1)
      logits = self.head(pooled)
      return logits
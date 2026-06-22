import torch
import torch.nn as nn

class ClassifierHead(nn.Module):

   def __init__(self):
      
      super().__init__()
      self.fc = nn.Sequential(nn.Linear(768,256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256,2))
   
   def forward(self, x):
      return self.fc(x)
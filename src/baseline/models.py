from torchvision import models
import torch.nn as nn

class FeatureExtractorResNet(nn.Module):
  "backbone feature extractor"

  def __init__(self):
      super().__init__()
      self.feature_extractor = models.resnet50(pretrained=True)
      self.feature_extractor.fc = nn.Identity()
      self.name = "FeatureExtractor"
      self.to(
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      )
      
  def forward(self, x):
    feature_map = self.feature_extractor(x)
    return feature_map


class BaseLineClassifer(nn.Module):
  """
  Baseline classifier composed by a simple Linear Layer with Dropout.

  Input arguments:
      dropout: float (.15) = probability of dropout.
      in_features_head: int (2048) = number of features from the feature extractor.
      num_classes: int (20) = dimension of final output.

  """

  def __init__(
      self, 
      dropout: float =.15,
      in_features_head: int = 2048,
      num_classes: int = 20
      ) -> None:

      super().__init__()

      self.classifier = nn.Sequential(
          nn.Dropout(dropout),
          nn.Linear(in_features_head, num_classes)
      )

      self.name = "BaseLineClassifier"
      self.to(
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      )

  def forward(self, x):
    output = self.classifier(x)
    return output
      

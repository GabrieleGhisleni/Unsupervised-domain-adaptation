from torchvision import models
import torch.nn as nn


class BottleNeckFeatureExtractor(nn.Module):
    """
    On the original paper they have used a CNN + three fully connected layers of 4096, 
    4096, n_classes ad added a lower dimensional 'bottleneck', adaptation layer.
    Here we used a ResNet + FFL {2056, 2056, 512, n_classes} and we will compute
    the MMD on the output of the 512 layer.
    
    Input arguments:
      dropout: float (.15) = probability of dropout.
      out_dim: int (512) = feature dimension out.
    """

    def __init__(
          self, 
          dropout: float = .15, 
          out_dim: int = 512
        ) -> None:

        super().__init__()
        self.feature_extractor = models.resnet50(pretrained=True)
        in_features_head = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        # heavy linear layers 
        self.dense_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features_head, in_features_head), 
            nn.Dropout(dropout), 
            nn.ReLU(),
            nn.Linear(in_features_head, in_features_head),
            nn.Dropout(dropout), 
            nn.ReLU(),
        )

        # bottle neck linear layers 
        self.bottle_neck = nn.Sequential(
            nn.Linear(in_features_head, out_dim), 
            nn.Dropout(dropout), 
            # nn.Tanh(), trial
            nn.ReLU(),
        )
        self.name = "BottleNeckFeatureExtractor"
        self.to(
              torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        
    def forward(self, x):
      feature_map = self.feature_extractor(x)
      dense_representation = self.dense_layer(feature_map)
      shrinked_feature_map = self.bottle_neck(dense_representation)
      return shrinked_feature_map


class MMDClassifer(nn.Module):
    """
    MMD classification headclassifier composed by a simple Linear Layer.
    the features that comes in are already passed through an activation function.

    Input arguments:
        dropout: float (.15) = probability of dropout.
        out_dim: int (512) = feature in from the feature_extractor.
        num_classes: int (20) = dimension of final output.
    """

    def __init__(
          self, 
          dropout: float = .15, 
          out_dim: int = 512, 
          num_classes: int = 20
        ) -> None:

        super().__init__()

        self.classifier_head = nn.Linear(out_dim, num_classes)
        self.name = "MMDClassifer"
        self.to(
              torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

    def forward(self, shrinked_feature_map):
      output = self.classifier_head(shrinked_feature_map)
      return output


class FeatureExtractorResNet(nn.Module):
    "Basic ResNet as backbone for feature extraction"

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


class RevGradClassifer(nn.Module):
    """
    Reversal gradient classification head composed by 4 linear layer 
    {2048|2048|512|n_classes} with dropout and ReLU activation function.

    Input arguments:
        dropout: float (.15) = probability of dropout.
        in_features_head: int (2048) = dimension of the vector provided by the feature extractor.
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
            nn.Linear(in_features_head, in_features_head), 
            nn.Dropout(dropout), 
            nn.ReLU(),
            nn.Linear(in_features_head, in_features_head), 
            nn.Dropout(dropout), 
            nn.ReLU(),
            nn.Linear(in_features_head, 512),
            nn.Dropout(dropout), 
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.name = "RevGradClassifer"
        self.to(
              torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

    def forward(self, x):
      output = self.classifier(x)
      return output
      

class RevGradDiscriminator(nn.Module):
    """
    Gradient Reversal Layer
    We reverse the gradient and we pass to a linear layer {2048 | 512 | 2}
    where the output is the discrimination between one domain or the other. 

    Input arguments:
        dropout: float (.15) = probability of dropout.
        in_features_head: int (2048) = dimension of the vector provided by the feature extractor.
        num_classes: int (20) = dimension of final output.
    """

    def __init__(
        self, 
        alpha: float = 1, 
        dropout: float = .15, 
        in_features_head = 2048
        ) -> None:

        super().__init__()

        self.discriminator = nn.Sequential(
            GradientReverse(alpha),
            nn.Linear(in_features_head, 512), 
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        self.name = "RevGradResNet"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
      return self.discriminator(x)

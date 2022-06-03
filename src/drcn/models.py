from utils import LogSize, FlattenToChannelView, ChannelToFlatView

################################################################################
# Keep as backbone the resnet50.

class DrcnBackBone_v14(nn.Module):
    "backbone feature extractor"

    def __init__(self):
      super().__init__()

      self.feature_extractor = models.resnet50(pretrained=True)
      self.feature_extractor.fc = nn.Identity()

      self.name = "DRCN_feature_extractor"
      self.to(
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      )
        
    def forward(self, x):
      feature_map = self.feature_extractor(x)
      return feature_map


class DrcnDecoder_v14(nn.Module):
    """
    Decoder architecture for reconstructing the image.
    trough the usage of deconvolutional layer, batch norm and upsamples.

    Input arguments:
      feature_extractor: nn.Module = nn for feature extraction
      c: int = number of channels
      w: int = width of the image
      h: int = height of the image
      test: bool = verbose to understand the image sizes.
    """

    def __init__(
        self, 
        feature_extractor: nn.Module, 
        c: int = 256, 
        w: int = 5, 
        h: int = 5, 
        test: bool = True
      ) -> None:

      super().__init__()

      self.feature_extractor = feature_extractor

      self.decoder = nn.Sequential(
          nn.Linear(2048, c * h * w),
          FlattenToChannelView(c, h, w),
          nn.ReLU(),

          nn.Upsample(scale_factor=2),
          nn.ConvTranspose2d(256, 128, kernel_size=(4, 4)),
          nn.BatchNorm2d(128),
          nn.ReLU(),

          nn.Upsample(scale_factor=2),
          nn.ConvTranspose2d(128, 64, kernel_size=(4, 4)),
          nn.BatchNorm2d(64),
          nn.ReLU(),

          nn.Upsample(scale_factor=2),
          nn.ConvTranspose2d(64, 32, kernel_size=(5, 5)),
          nn.BatchNorm2d(32),
          nn.ReLU(),

          nn.Upsample(scale_factor=2),
          nn.ConvTranspose2d(32, 3, kernel_size=(5, 5)),
          nn.Sigmoid(),
          LogSize('decoder out', test),
      )

      self.name = "DRCN_decoder"
      self.to(
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      )
      
    def forward(self, x):
      latent_feature = self.feature_extractor(x)
      rec_img = self.decoder(latent_feature)
      return rec_img


class DrcnHead_v14(nn.Module):
    """
    Drcn classification head

    Input arguments:
        feature_extractor: nn.Module = feature extractor
        dropout: float (.15) = probability of dropout.
        num_classes: int (20) = dimension of final output.
    """

    def __init__(
        self, 
        feature_extractor: nn.Module, 
        dropout:float = .15, 
        num_classes: int = 20
        ) -> None:

        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes), 
        )

        self.name = "DRCN_head"
        self.to(
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

    def forward(self, x):
      map_feature = self.feature_extractor(x)
      output = self.classifier_head(map_feature)
      return output

################################################################################
# Replacing ResNet with a conv net trained from scratch.

class DrcnEncoder_v10(nn.Module):
    """
    Encoder architecture for extract features from the images.
    trough the usage of convolutional layer, batch norm and max pooling.

    Input arguments:
      feature_extractor: nn.Module = nn for feature extraction.
      c: int = number of channels.
      w: int = width of the image.
      h: int = height of the image.
      test: bool = verbose to understand the image sizes.
    """

    def __init__(
        self, 
        c: int = 256, 
        w: int = 5, 
        h: int = 5, 
        test: bool = True
      ) -> None:

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(4, 4)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=(4, 4)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.ReLU(),

            ChannelToFlatView(c*w*h),
            LogSize(test=test, txt='encoder out'),
        )


        self.latent = nn.Sequential(
            nn.Linear(c * h * w, 1024),
            nn.ReLU()
        )


        self.name = "DRCN decoder v10"
        self.to(
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

    def forward(self, x):
      feature_map = self.encoder(x)
      latent_feature = self.latent(feature_map)
      return latent_feature


class DrcnDecoder_v10(nn.Module):
    """
    Decoder architecture for reconstructing the image.
    trough the usage of deconvolutional layer, batch norm and upsamples.

    Input arguments:
      feature_extractor: nn.Module = nn for feature extraction
      c: int = number of channels
      w: int = width of the image
      h: int = height of the image
      test: bool = verbose to understand the image sizes.

    """

    def __init__(
        self, 
        feature_extractor: nn.Module, 
        c: int = 256, 
        w: int = 5, 
        h: int = 5, 
        test: bool = True
      ) -> None:

      super().__init__()

      self.feature_extractor = feature_extractor
      self.decoder = nn.Sequential(
          nn.Linear(1024, c * h * w),
          FlattenToChannelView(c, h, w),
          nn.ReLU(),

          nn.Upsample(scale_factor=2),
          nn.ConvTranspose2d(256, 128, kernel_size=(4, 4)),
          nn.BatchNorm2d(128),
          nn.ReLU(),

          nn.Upsample(scale_factor=2),
          nn.ConvTranspose2d(128, 64, kernel_size=(4, 4)),
          nn.BatchNorm2d(64),
          nn.ReLU(),

          nn.Upsample(scale_factor=2),
          nn.ConvTranspose2d(64, 32, kernel_size=(5, 5)),
          nn.BatchNorm2d(32),
          nn.ReLU(),

          nn.Upsample(scale_factor=2),
          nn.ConvTranspose2d(32, 3, kernel_size=(5, 5)),
          nn.Sigmoid(),
          LogSize(test=test, txt='decoder out'),
      )
      self.name = "DRCN encoder v10"
      self.to(
          torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      )
        
    def forward(self, x):
      feature = self.feature_extractor(x)
      rec_img = self.decoder(feature)
      return rec_img


class DrcnHead_v10(nn.Module):
    """
    Drcn classification head

    Input arguments:
        feature_extractor: nn.Module = feature extractor
        dropout: float (.15) = probability of dropout.
        num_classes: int (20) = dimension of final output.
    """
    def __init__(
        self, 
        feature_extractor: nn.Module, 
        dropout:float = .15, 
        num_classes: int = 20
        ) -> None:

        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes), 
        )

        self.name = "DRCN Head v10"
        self.to(
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )


    def forward(self, x):
      map_feature = self.feature_extractor(x)
      output = self.classifier_head(map_feature)
      return output
class RevGradTrainer(BasicTrainer):
  """
  Reversal Gradient trainer.

  Input arguments:
      classifier: nn.Module = nn for the classification. 
      discriminator: nn.Module = nn for discriminate the source from the target domain.
      feature_extractor: nn.Module = nn for feature extractor (if incorporate into the classifier set to None).
      optimizer: torch.optim.Optimizer = optimizer for training the model.
      scheduler: torch.optim.lr_scheduler.ExponentialLR = scheduler for shrinking the lr during the training.
      patience: int = number of epochs without improving before being stopped.
      epochs: int = number of epochs to be performed.
  """

  def __init__(
      self, 
      classifier: nn.Module, 
      discriminator: nn.Module,
      feature_extractor: nn.Module, 
      optimizer: torch.optim.Optimizer, 
      scheduler: torch.optim.lr_scheduler.ExponentialLR, 
      patience: int, 
      epochs: int, 
    ) -> None:

    self.discriminator = discriminator
    self.optimizer = optimizer
    self.scheduler = scheduler

    self.extra_saves = dict(
        classifier=classifier,
        feature_extractor=feature_extractor,
        discriminator=discriminator,
        scheduler=scheduler,
        optimizer=optimizer
        )
    
    super().__init__(patience=patience, epochs=epochs, **self.extra_saves)


  def classifier_step(
      self, 
      feature_map: Tensor, 
      labels: Tensor
    ) -> Tensor:

    outputs = self.classifier(feature_map)
    classifier_loss = self.cross_entropy(outputs, labels)
    return classifier_loss


  def discriminator_step(
      self, 
      map_features, 
      labels
    ) -> None:

    """ 
    Discriminator step performed between the source and targer domain. 
    Input arguments:
      map_features: Tensor = feture map obtained from the feature extractor
      labels: Tensor = ground truth
    Return:
      Tensor = cross entropy loss between the prediction and the ground truth.
    """

    outputs = self.discriminator(map_features)
    discriminator_loss = self.cross_entropy(outputs, labels)
    return discriminator_loss


  def train(
        self, 
        training_loader: DataLoader, 
        testing_loader: DataLoader
      ) -> None:

      progress_bar_epoch = trange(1, self.epochs, leave=True, desc="Epoch")
      self.patienting = 0
      self.best_accuracy = 0 
      self.previous_epoch_loss = 1e5

      for epoch in progress_bar_epoch:
          self.classifier.train()
          self.feature_extractor.train()
          self.discriminator.train()
          total_classification_loss, n_samples = 0, 0
          total_discriminator_target_loss, total_discriminator_source_loss = 0, 0
          progress_bar_batch = tqdm(enumerate(training_loader), leave=False, total=len(training_loader), desc="Training")

          for idx, data in progress_bar_batch:
              self.optimizer.zero_grad()
              source_images = data["source_domain_image"].to(self.device, dtype=torch.float)
              target_images = data["target_domain_image"].to(self.device, dtype=torch.float)
              labels = data["source_domain_label"].to(self.device, dtype=torch.long)

              batch_size = source_images.size(0)

              # ground truth for the discriminator step
              zeros_label = torch.zeros(batch_size, dtype=torch.long, device=self.device)
              ones_label = torch.ones(batch_size, dtype=torch.long, device=self.device)

              shrinked_feature_map_source = self.feature_extractor(source_images)
              shrinked_feature_map_target = self.feature_extractor(target_images)

              discriminator_source_loss = self.discriminator_step(shrinked_feature_map_source, zeros_label)
              discriminator_target_loss = self.discriminator_step(shrinked_feature_map_target, ones_label)

              classification_loss = self.classifier_step(shrinked_feature_map_source, labels)

              loss = classification_loss + discriminator_target_loss + discriminator_source_loss
              loss.backward()
              self.optimizer.step()

              total_classification_loss += classification_loss.item()
              total_discriminator_source_loss += discriminator_source_loss.item()
              total_discriminator_target_loss += discriminator_target_loss.item()
              n_samples += batch_size

              loss_progress = {
                  "classifier_loss": total_classification_loss/n_samples, 
                  "discriminator_source_loss": total_discriminator_source_loss/n_samples,
                  "discriminator_target_loss": (total_discriminator_target_loss)/n_samples,
                  "total_loss": (total_classification_loss + total_discriminator_source_loss + total_discriminator_target_loss) / n_samples
                  }

              if (idx % 2 == 0 and idx):
                  progress_bar_batch.set_postfix(loss_progress)

          if self.scheduler:
            self.scheduler.step()

          wandb.log(loss_progress, commit=False)
          validation_metrics = self.validation(testing_loader, epoch)
          progress_bar_epoch.set_postfix(validation_metrics)


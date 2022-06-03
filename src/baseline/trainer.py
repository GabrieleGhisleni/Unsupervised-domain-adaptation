class BaselineTrainer(BasicTrainer):
  """
  Baseline trainer.

  Input arguments:
      classifier: nn.Module = nn for the classification. 
      feature_extractor: nn.Module = nn for feature extractor (if incorporate into the classifier set to None).
      optimizer: torch.optim.Optimizer = optimizer for training the model.
      scheduler: torch.optim.lr_scheduler.ExponentialLR = scheduler for shrinking the lr during the training.
      patience: int = number of epochs without improving before being stopped.
      epochs: int = number of epochs to be performed.

  """
  
  def __init__(
      self, 
      classifier: nn.Module, 
      feature_extractor: nn.Module, 
      optimizer: torch.optim.Optimizer, 
      scheduler: torch.optim.lr_scheduler.ExponentialLR, 
      patience: int, 
      epochs: int
    ) -> None:

    self.optimizer = optimizer
    self.scheduler = scheduler

    self.extra_saves = dict(
      classifier=classifier,
      feature_extractor=feature_extractor,
      optimizer=optimizer,
      scheduler=scheduler    
    )

    super().__init__(patience=patience, epochs=epochs, **self.extra_saves)
  
  def classifier_step(
      self, 
      batch: dict
    ) -> dict:

    """
    Classification step performed on the source domain. 
    
    Input arguments:
      batch: dict = batch from custom dataloaders.

    Return:
      dict = cross entropy loss between the prediction and the ground truth.
    """
    
    image = batch["source_domain_image"].to(self.device, dtype=torch.float)
    targets = batch["source_domain_label"].to(self.device, dtype=torch.long)
    feature_map = self.feature_extractor(image)
    outputs = self.classifier(feature_map)
    return self.cross_entropy(outputs, targets)

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
          self.feature_extractor.train()
          self.classifier.train()
          total_train_classifier_loss, n_samples = 0, 0
          progress_bar_batch = tqdm(enumerate(training_loader), leave=False, total=len(training_loader), desc="Training")

          for idx, data in progress_bar_batch:
              self.optimizer.zero_grad()
              loss = self.classifier_step(data)
              loss.backward()
              self.optimizer.step()

              total_train_classifier_loss += loss.item()
              n_samples += data["source_domain_image"].size(0)

              if (idx % 5 == 0 and idx):
                  progress_bar_batch.set_postfix({"classifier_loss": total_train_classifier_loss/n_samples})

          if self.scheduler:
            self.scheduler.step()
            
          wandb.log({"classifier_loss": total_train_classifier_loss/n_samples}, commit=False)
          validation_metrics = self.validation(testing_loader, epoch)
          progress_bar_epoch.set_postfix(validation_metrics)
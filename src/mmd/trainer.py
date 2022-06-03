class MMDTrainer(BasicTrainer):
  """
  MMD trainer.

  Input arguments:
      classifier: nn.Module = nn for the classification. 
      feature_extractor: nn.Module = nn for feature extractor (if incorporate into the classifier set to None).
      optimizer: torch.optim.Optimizer = optimizer for training the model.
      scheduler: torch.optim.lr_scheduler.ExponentialLR = scheduler for shrinking the lr during the training.
      patience: int = number of epochs without improving before being stopped.
      epochs: int = number of epochs to be performed.
      mmd_lambda: float (.25) = parameter to regulate the tradeoff between classification and mmd loss.
      kernel: str ("multiscale") = kernel used for mmd.

  """

  def __init__(
      self, 
      classifier: nn.Module, 
      feature_extractor: nn.Module, 
      optimizer: torch.optim.Optimizer, 
      scheduler: torch.optim.lr_scheduler.ExponentialLR, 
      patience: int, 
      epochs: int, 
      mmd_lambda: float = .25, 
      kernel: str = "multiscale"
      ) -> None:

    self.optimizer = optimizer
    self.scheduler = scheduler
    self.mmd_lambda = mmd_lambda
    self.kernel = kernel

    self.extra_saves = dict(
      classifier=classifier,
      feature_extractor=feature_extractor,
      optimizer=optimizer,
      scheduler=scheduler    
    )

    super().__init__(patience=patience, epochs=epochs, **self.extra_saves)
  

  def classifier_step(
      self, 
      shrinked_feature_map_source, 
      labels
    ) -> Tensor:

    outputs = self.classifier(shrinked_feature_map_source)
    classifier_loss = self.cross_entropy(outputs, labels)
    return classifier_loss


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
          total_classifier_loss, mmd_total_loss, n_samples = 0, 0, 0
          progress_bar_batch = tqdm(enumerate(training_loader), leave=False, total=len(training_loader), desc="Training")

          for idx, data in progress_bar_batch:
              self.optimizer.zero_grad()

              source_images = data["source_domain_image"].to(self.device, dtype=torch.float)
              target_images = data["target_domain_image"].to(self.device, dtype=torch.float)
              labels = data["source_domain_label"].to(self.device, dtype=torch.long)

              # mmd step
              shrinked_feature_map_source = self.feature_extractor(source_images)
              shrinked_feature_map_target = self.feature_extractor(target_images)

              mmd_loss = self.maximum_mean_discrepancies(
                    shrinked_feature_map_source, 
                    shrinked_feature_map_target, 
                    self.kernel
                  )
              
              mmd_loss_adjusted = (self.mmd_lambda * mmd_loss)
              
              classification_loss = self.classifier_step(shrinked_feature_map_source, labels)

              loss = classification_loss + mmd_loss_adjusted
              loss.backward()
              self.optimizer.step()

              total_classifier_loss += classification_loss.item()
              mmd_total_loss += mmd_loss.item()
              n_samples += data["source_domain_image"].size()[0]

              loss_progress = {
                  "classifier_loss": total_classifier_loss/n_samples, 
                  "mmd_loss": mmd_total_loss/n_samples,
                  "total_loss": (total_classifier_loss+mmd_total_loss)/n_samples
                  }

              if (idx % 2 == 0 and idx):
                progress_bar_batch.set_postfix(loss_progress)

          if self.scheduler:
            self.scheduler.step()

          wandb.log(loss_progress, commit=False)
          validation_metrics = self.validation(testing_loader, epoch)
          progress_bar_epoch.set_postfix(validation_metrics)


  def maximum_mean_discrepancies(self, x, y, kernel="multiscale"):
      """
      # https://www.kaggle.com/code/onurtunali/maximum-mean-discrepancy/notebook
      Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
      """
      xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
      rx = (xx.diag().unsqueeze(0).expand_as(xx))
      ry = (yy.diag().unsqueeze(0).expand_as(yy))
      
      dxx = rx.t() + rx - 2. * xx # Used for A in (1)
      dyy = ry.t() + ry - 2. * yy # Used for B in (1)
      dxy = rx.t() + ry - 2. * zz # Used for C in (1)
      
      XX, YY, XY = (torch.zeros(xx.shape).to(device),
                    torch.zeros(xx.shape).to(device),
                    torch.zeros(xx.shape).to(device))
      
      if kernel == "multiscale":
          bandwidth_range = [0.2, 0.5, 0.9, 1.3]
          for a in bandwidth_range:
              XX += a**2 * (a**2 + dxx)**-1
              YY += a**2 * (a**2 + dyy)**-1
              XY += a**2 * (a**2 + dxy)**-1
              
      if kernel == "rbf":
          bandwidth_range = [10, 15, 20, 50]
          for a in bandwidth_range:
              XX += torch.exp(-0.5*dxx/a)
              YY += torch.exp(-0.5*dyy/a)
              XY += torch.exp(-0.5*dxy/a)
  
      return torch.mean(XX + YY - 2. * XY)
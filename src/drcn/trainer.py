class DrcnTrainer(BasicTrainer):
  """
  Deep Reconstruction Classifier Network.
  Since we had many difficulties  implementing this method we tried different
  solution and exception while trying as keeping the backbone freeze for a certain
  amount of epoch and start training only the decoder; reconstructing both the target
  and the source domain.

  Input arguments:
      classifier: nn.Module = nn for the classification. 
      classifier_optimizer: torch.optim.Optimizer = optimizer for training the classifier 
      classifier_scheduler: torch.optim.lr_scheduler.ExponentialLR = scheduler for shrinking the lr of the classifier during the training.
      generator: nn.Module = nn for the encoding. 
      generator_optimizer: torch.optim.Optimizer =  optimizer for training the encoder 
      generator_scheduler: : torch.optim.lr_scheduler.ExponentialLR = scheduler for shrinking the lr of the encoder during the training.
      patience: int = number of epochs without improving before being stopped.
      epochs: int = number of epochs to be performed.

      alpha: float = parameter for controlling the importance of the reconstruction loss
      reconstruct_source: bool = to allow reconstruction the source domain together with the target domain
      train_classifier: bool = to keep the classifier freezed
      unfreeze_backbone: int = epoch to unfreeze the backbone
      lr_backbone: float = learning rate of the backbone after the unfreezing
  """
  def __init__(
      self, 
      classifier: nn.Module, 
      classifier_optimizer: torch.optim.Optimizer, 
      classifier_scheduler: torch.optim.lr_scheduler.ExponentialLR,
      generator: nn.Module, 
      generator_optimizer: torch.optim.Optimizer, 
      generator_scheduler: torch.optim.lr_scheduler.ExponentialLR,
      patience: int, 
      epochs: int,
      alpha: float,
      reconstruct_source: bool,
      train_classifier: bool,
      unfreeze_backbone: int,
      lr_backbone: float
    ) -> None:
    
    self.generator = generator
    self.generator_optimizer = generator_optimizer
    self.generator_scheduler = generator_scheduler
    self.classifier_optimizer = classifier_optimizer
    self.classifier_scheduler = classifier_scheduler

    self.alpha = alpha
    # self.mse_loss = nn.MSELoss() # (reduction='sum')
    self.l1_loss = nn.L1Loss() # (reduction='sum')

    self.reconstruct_source = reconstruct_source
    self.train_classifier = train_classifier
    self.unfreeze_backbone = unfreeze_backbone
    self.lr_backbone = lr_backbone

    self.extra_saves = dict(
        generator=generator,
        generator_optimizer=generator_optimizer,
        generator_scheduler=generator_scheduler,
        classifier=classifier,
        classifier_optimizer=classifier_optimizer,
        classifier_scheduler=classifier_scheduler,
        feature_extractor=None, 
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


  def generator_step(self, images: Tensor) -> Tensor:
    """ 
    encoder-decoder step 
    Input arguments: images: Tensor = feture map obtained from the feature extractor
    Return:
      Tensor = l1loss or mse loss between the reconstructed image and the original.
    """
    regenerated_image = self.generator(images)
    generator_loss = self.l1_loss(regenerated_image,  images)
    return generator_loss


  def train(self, train: DataLoader, test: DataLoader) -> None:
    progress_bar_epoch = trange(1, self.epochs, leave=True, desc="Epoch")
    self.patienting = 0
    self.best_accuracy = 0 
    self.previous_epoch_loss = 1e5

    for epoch in progress_bar_epoch:
        if self.unfreeze_backbone == epoch:
          self.unfreeze()
          print(f"unfreezing at epoch: {epoch}\nTrainable layers:")
          for n, p in self.generator.named_parameters():
            if p.requires_grad:
              print(n)

        total_generator_source_loss, total_generator_target_loss, total_classification_loss, n_samples = (0, 0, 0, 0)
        progress_bar_batch = tqdm(enumerate(train), leave=False, total=len(train), desc="Training reconstructor")
        self.generator.train()
        loss_progress = {}

        # Generator step
        for idx, data in progress_bar_batch:
            n_samples += data["target_domain_image"].size(0)
            self.generator_optimizer.zero_grad()
            self.generator.zero_grad()

            if self.reconstruct_source:
              target_images = data["target_domain_image"].to(self.device, dtype=torch.float)
              source_images = data["source_domain_image"].to(self.device, dtype=torch.float)
              generator_target_loss = self.generator_step(target_images)
              generator_source_loss = self.generator_step(source_images)
              total_loss = self.alpha * (generator_target_loss + generator_source_loss)
              total_loss.backward()
              self.generator_optimizer.step()
              
              total_generator_source_loss += generator_source_loss.item()
              loss_progress["generator_source_loss"] = (total_generator_source_loss)/n_samples

            else:
              target_images = data["target_domain_image"].to(self.device, dtype=torch.float)
              generator_target_loss = self.generator_step(target_images)
              generator_target_loss = self.alpha * generator_target_loss
              generator_target_loss.backward()
              self.generator_optimizer.step()
              
            total_generator_target_loss += generator_target_loss.item()
            loss_progress["generator_target_loss"] = (total_generator_target_loss)/n_samples
            progress_bar_batch.set_postfix(loss_progress)

        n_samples = 0
        self.generator.zero_grad()
        self.show_images_progress(data, epoch)
        wandb.log(loss_progress, commit=False)

        if self.generator_scheduler:
          self.generator_scheduler.step()

        # Classification step
        if self.train_classifier:
          self.classifier.train()
          progress_bar_batch = tqdm(enumerate(train), leave=False, total=len(train), desc="Training classifier")
          for idx, data in progress_bar_batch:
              self.classifier_optimizer.zero_grad()

              source_images = data["source_domain_image"].to(self.device, dtype=torch.float)
              labels = data["source_domain_label"].to(self.device, dtype=torch.long)
              classification_loss = self.classifier_step(source_images, labels)
              classification_loss.backward()
              self.classifier_optimizer.step()

              n_samples += source_images.size(0)
              total_classification_loss += classification_loss.item()
              loss_progress["classifier_loss"] = total_classification_loss/n_samples
              progress_bar_batch.set_postfix(loss_progress)
        
          if self.classifier_scheduler:
            self.classifier_scheduler.step()

          wandb.log(loss_progress, commit=False)
          validation_metrics = self.validation(test, epoch)
          progress_bar_epoch.set_postfix(validation_metrics)


  def show_images_progress(self, data: dict, e: int):
    with torch.no_grad():
      imgs = self.generator(data['target_domain_image'][0:4].to(self.device))

    fix, axs = plt.subplots(ncols=4, nrows=2, squeeze=False, figsize=(15,10))
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    axs[1,0].imshow(T.ToPILImage()(data['target_domain_image'][0].to('cpu')))
    axs[1,1].imshow(T.ToPILImage()(data['target_domain_image'][1].to('cpu')))
    axs[1,2].imshow(T.ToPILImage()(data['target_domain_image'][2].to('cpu')))
    axs[1,3].imshow(T.ToPILImage()(data['target_domain_image'][3].to('cpu')))

    plt.xlabel(f"Epoch number: {e}")
    plt.tight_layout()
    wandb.log({"chart": plt})
    plt.show()

  def unfreeze(self):
    f_params = []
    for name, par in self.classifier.named_parameters():
      par.requires_grad = True
      if 'feature' in name and 'latent' not in name:
        f_params.append(par)

    self.generator_optimizer.add_param_group({'params': f_params, 'lr': self.lr_backbone}) 
    self.classifier_optimizer.add_param_group({'params': f_params,  'lr': self.lr_backbone}) 

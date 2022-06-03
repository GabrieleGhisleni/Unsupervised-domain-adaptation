from tqdm.notebook import tqdm, trange
from sklearn import metrics
from torch import Tensor
import datetime as dt
from torch import nn

class BasicTrainer:
    """
    BasicTrainer shared across all the experiments for standard functionalities

    Input arguments:
        classifier: nn.Module = nn for the classification. 
        feature_extractor: nn.Module = nn for feature extractor (if incorporate into the classifier set to None).
        patience: int = number of epochs without improving before being stopped.
        epochs: int = number of epochs to be performed.

    """

    def __init__(
        self, 
        classifier: nn.Module, 
        feature_extractor: nn.Module, 
        patience: int, 
        epochs: int, 
        **kwargs
        ) -> None:

        self.extra_saves = kwargs
        self.epochs = epochs
        self.patience = patience
        self.classifier = classifier

        self.feature_extractor = feature_extractor
        self.cross_entropy = nn.CrossEntropyLoss()
         
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.classes_to_idx = { 
            'backpack': 0,
            'bookcase': 1,
            'car jack': 2,
            'comb': 3,
            'crown': 4,
            'file cabinet': 5,
            'flat iron': 6,
            'game controller': 7,
            'glasses': 8,
            'helicopter': 9,
            'ice skates': 10,
            'letter tray': 11,
            'monitor': 12,
            'mug': 13,
            'network switch': 14,
            'over-ear headphones': 15,
            'pen': 16,
            'purse': 17,
            'stand mixer': 18,
            'stroller': 19
        }

        self.idx_to_class = {v: k for k, v in self.classes_to_idx.items()}

    def training_loop(
        self, 
        training_loader: DataLoader, 
        testing_loader: DataLoader, 
        wandb_tags: list = None, 
        wandb_run_id: str = None, 
        wandb_run_name: str = None, 
        **kwargs
      ) -> None:

        """
        Wrapper function for logging into wandb the details of the experiment

        Input arguments:
            training_loader: DataLoader = dataloader used to training.
            testing_loader: DataLoader = dataloader used to testing.
            wandb_tags: list = tags to be added into the wandb run, 
            wandb_run_name: str = name of the run, 
            wandb_run_id: str = specify the id if there is the need to restore a run, 

        """

        wandb.init(
            project="UDA",
            entity="ggabry",
            config=kwargs,
            id=wandb_run_id,
            tags=wandb_tags,
            name=wandb_run_name,
            settings=wandb.Settings(symlink=False),
        )

        wandb.config.wandb_run_dir = str(wandb.run.dir).split("____SHARED")[-1]

        try: 
          self.train(training_loader, testing_loader)
        except KeyboardInterrupt: 
          print(f"Processed termined by the user at: {self.get_time()}")
        except SystemExit: 
          print(f"Processed termined by EalyStopping at: {self.get_time()}")
        finally: 
          wandb.finish()


    def validation_step(
        self, 
        batch: dict
      ) -> Tuple[Tensor, Tensor, Tensor]:
      
      """
      Extract the feature, classify and compute the cross entropy loss between
      the ground truth and the prediction.

      Input arguments:
          batch: dict = batch from custom dataloader.
        
      Return:
        Tuple[
          output: Tensor = model prediction. 
          targets: Tensor = ground truth.
          loss: Tensor = cross entropy loss.
        ]
      """
      
      image = batch["target_domain_image"].to(self.device, dtype=torch.float)
      targets = batch["target_domain_label"].to(self.device, dtype=torch.long)

      if self.feature_extractor is None:
        outputs = self.classifier(image)

      else:
        feature_map = self.feature_extractor(image)
        outputs = self.classifier(feature_map)

      loss = self.cross_entropy(outputs, targets)
      return outputs, targets, loss


    def validation(
        self, 
        testing_loader: DataLoader, 
        epoch: int
      ) -> dict:
      
      """
      Validate all the test set, compute the accuracy, store the total test loss 
      and check for the improvement in the performance; if it is the case, save the
      new model checkpoint.
      
      Input arguments:
        testing_loader: DataLoader = testing custom dataloader
        epoch: int = current epoch
        
      Return:
        dict = dictionary containing the relevant metrics.

      """

      if self.feature_extractor:
        self.feature_extractor.eval()

      self.classifier.eval()
      with torch.no_grad():
          total_test_classifer_loss, total_preds = (0, 0)
          predictions, labels = ([], [])
          progress_bar_test = tqdm(enumerate(testing_loader), leave=False, total=len(testing_loader), desc="Testing")
          for idx, data in progress_bar_test:
            outputs, targets, loss = self.validation_step(data)
            total_test_classifer_loss += loss.item()

            #class level metrics
            _, predicted = torch.max(outputs.data, axis=1)
            predictions.extend(predicted.cpu())
            labels.extend(targets.cpu())
            total_preds += targets.size(0)


      #compute metrics
      test_classifer_loss = total_test_classifer_loss / total_preds
      to_update_tqdm_bar, evals = self.compute_test_metrics(labels, predictions, 
                                                            test_classifer_loss, 
                                                            epoch)
      # check if improvement and earlystopping
      self.check_earlystopping_and_improvement(evals,
                                              test_classifer_loss, 
                                              epoch)          
      return to_update_tqdm_bar


    def compute_test_metrics(
        self, 
        labels: List[int], 
        predictions: List[int], 
        test_classifer_loss: float, 
        epoch: int
      ) -> dict:

      """
      Compute the actual metrics for the current epoch
      
      Input arguments:
        labels: List[int] = ground truth
        predictions: List[int] = model prediction
        test_classifer_loss: float = total test loss
        epoch: int = current epoch
        
      Return:
        dict = dictionary containing the relevant metrics.

      """

      target_names = [self.idx_to_class[i] for i in range(20)]
      evals = metrics.classification_report(
          labels, predictions, 
          output_dict=True, 
          zero_division=0, 
          target_names=target_names
        )

      evals["current epoch"] = epoch
      evals["total_test_loss"] = (test_classifer_loss)
      wandb.log(evals)

      to_update_tqdm_bar = dict(
          epoch = epoch,
          current_test_loss = test_classifer_loss,
          current_accuracy = evals["accuracy"],
          best_accuracy_so_far = self.best_accuracy,
          best_test_loss_so_far = self.previous_epoch_loss,  
      )
          
      return (to_update_tqdm_bar, evals)


    def check_earlystopping_and_improvement(
        self, 
        evals: dict, 
        test_classifer_loss: float, 
        epoch: int
      ) -> None:

      """
      Check if the model has improved in the during the training of the last epoch.
      If is not the case update the counter of patience and if the max patience 
      is reached stop the training process.
      
      Input arguments:
        evals: dict = metrics for the current epoch.
        test_classifer_loss: float = total test loss for the current epoch.
        epoch: int = current epoch.

      """

      # check patience reached
      if self.patience <= self.patienting:
        sys.exit()

      # looking for improvemenent
      if (
          test_classifer_loss < self.previous_epoch_loss 
          or self.best_accuracy < evals["accuracy"]
      ):
          self.patienting = 0
          self.save_models_and_params(epoch)

          #further logs and update
          if test_classifer_loss < self.previous_epoch_loss:
            self.previous_epoch_loss = test_classifer_loss
            wandb.log({'best_test_loss': self.previous_epoch_loss})

          if self.best_accuracy < evals["accuracy"]:
            self.best_accuracy = evals["accuracy"]
            wandb.log({'best_accuracy': self.best_accuracy})

      else:
        self.patienting += 1
        print(f"Skip checkpoint at  epoch {epoch} | time: {self.get_time()}")


    def save_models_and_params(
        self, 
        epoch: int
      ) -> None:

      """
      Save the model and all the relevant objects.       
      
      Input arguments:
        epoch: int = current epoch.

      """

      re_training_params = dict(
          classifier = self.classifier.state_dict(),
          feature_extractor = self.feature_extractor.state_dict() if self.feature_extractor else None,
          epoch = epoch
      )

      for extra_save, extra_value in self.extra_saves.items():
        re_training_params[extra_save] = extra_value.state_dict()

      torch.save(re_training_params, os.path.join(wandb.run.dir, f"{self.classifier.name}-re_train_args.pt"))
      wandb.save(f"{self.classifier.name}-re_train_args.pt")
      print(f"End of epoch: {epoch} | Saving checkpoint | at {self.get_time()}")


    def get_time(self):
      return dt.datetime.now().strftime('%H:%M:%S')

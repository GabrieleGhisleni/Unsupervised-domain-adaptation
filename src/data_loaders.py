from src.utils import show_values_on_bars, create_pandas_from_wandb
from src.utils import show, dotdict, check_grad_names
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from typing import Callable, Tuple
import torchvision.transforms as T
import torch.nn.functional as F
import random


class UdaDataset(Dataset):
    def __init__(
        self, 
        source_domain: ImageFolder, 
        target_domain: ImageFolder, 
        verbose: bool=False
      ) -> None:
        
      self.source_domain = source_domain
      self.target_domain = target_domain
      self.mapping = self.target_domain.class_to_idx
      assert (self.source_domain[0][0].shape == self.target_domain[0][0].shape)

      if verbose:
        print(f"source_domain root: {source_domain.root}")
        print(f"target_domain root: {target_domain.root}\n")
        print(f"transform function: {source_domain.transform}\n")
        print(f"\nimages shape: {self.source_domain[0][0].shape}")

    def __getitem__(self, idx) -> dict:
      return (
          {
              'source_domain_image': self.source_domain[idx][0],
              'source_domain_label': self.source_domain[idx][1],
              'target_domain_image': self.target_domain[idx][0], 
              'target_domain_label': self.target_domain[idx][1], 
          }
      )

    def __len__(self):
        return min([len(self.source_domain), len(self.target_domain)]) -1



def create_train_test(
    test_size: float, 
    transform_source: Callable,
    return_eval_df: bool = False, 
    transform_target: (Callable or None) = None,
    source_domain_path: str = "adaptiope_small/product_images",
    target_domain_path: str = "adaptiope_small/real_life",
    batch_size: int = 32,
    print_mapping: bool = False,
  ) -> Tuple[DataLoader, DataLoader]:
  """
  Create the DataLoader classes from the two root (source and target);

  Input arguments:
      test_size: float = proportion of test size, 
      batch_size: int = 32: number of images x batch.
      return_eval_df: bool = create a third dataloader for evaluating.
      print_mapping: bool = False: to inspect the mapping of the classes.
      target_domain_path: str = "adaptiope_small/real_life": target domain root path.
      source_domain_path: str = "adaptiope_small/product_images": source domain root path.
      transform_source: Callable = torchvision transform to apply on the source domain images.
      transform_target: (Callable or None) = torchvision transform to apply on the target domain images; if None use the same as the source.

  Return:
    Tuple[DataLoader, DataLoader]

  """

  if transform_target is None:
    transform_target = transform_source

  whole_dataset = UdaDataset(
      ImageFolder(source_domain_path, transform=transform_source),
      ImageFolder(target_domain_path, transform=transform_target),
      verbose=print_mapping
      )
  
  test_range = int(len(whole_dataset) * test_size)
  train_range = int(len(whole_dataset) - test_range)

  generator = torch.Generator().manual_seed(42) # for having always the same division (ensure reproducibility)
  subsets = random_split(whole_dataset,  [train_range, test_range], generator=generator)

  if print_mapping:
      print(f"class_to_idx: ")
      pprint(whole_dataset.mapping)

  return (
          DataLoader(subsets[0], batch_size=batch_size, shuffle=True, drop_last=False,  num_workers=2, pin_memory=True),
          DataLoader(subsets[1], batch_size=batch_size, shuffle=True, drop_last=False,  num_workers=2, pin_memory=True)
          )

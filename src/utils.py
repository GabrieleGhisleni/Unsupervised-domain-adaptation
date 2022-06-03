######### UTILS
from typing import Callable, Tuple
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch import Tensor
from torch import nn
import pandas as pd
import numpy as np
import pprint
import torch
import wandb


def show_values_on_bars(
    axs,
    h_v: str = "v",
    space_x: float = 0.12,
    space_y: float = 0.1,
    fontdict: dict = None,
    round_to: int = 0,
  ) -> None:
    """
    Show the values on the bar
    Parameters:
        axs: matplotlib axes with the barplot already displayed.
        h_v: str (v) = vertical or 'h' bars.
        space_x: float (0.) = adjust the values of bar x position
        space_y: float (0.) = adjust the values of bar y position
        fontdict: dict = dict for adjust the font of the values.
        round_to: int = round to N decimal (0 for int).

    Return:
        None
    """

    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2 + space_x
                _y = p.get_y() + p.get_height() + (space_y)
                if not np.isnan(p.get_width()):
                    value = round(p.get_height(), round_to)
                    if round_to == 0:
                        value = int(value)

                    if fontdict:
                        ax.text(_x, _y, value, ha="left", fontdict=fontdict)
                    else:
                        ax.text(_x, _y, value, ha="left")
        elif h_v == "h":
            for p in ax.patches:
                try:
                    _x = p.get_x() + p.get_width() + space_x
                    _y = p.get_y() + p.get_height() + space_y
                    if not np.isnan(p.get_width()):
                        value = round(p.get_width(), round_to)
                        if round_to == 0:
                            value = int(value)
                        if value < 0:
                            _x -= 0.27
                        if fontdict:
                            ax.text(_x, _y, value, ha="left", fontdict=fontdict)
                        else:
                            ax.text(_x, _y, value, ha="left")
                except:
                    print(f"Error while preparing {str(p)}")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def show(imgs: Tensor) -> None:
  "given n torch.Tensor show the corresponding image"
  fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
  for i, img in enumerate(imgs):
    img = T.ToPILImage()(img.to('cpu'))
    axs[0, i].imshow(np.asarray(img))
    axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class dotdict(dict):
  'fancy dotdict class for access dictionary with dot'
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  def print_all(self):
      pprint.PrettyPrinter(indent=4).pprint(self)


def create_pandas_from_wandb(
    runs, 
    direction='Product -> R'
  ) -> pd.DataFrame:
  "Fetch the data from wandb an create a nice pd tables."

  def check_exp(l):
    if 'GradRev' in l: return "GradRev"
    if 'mmd' in l: return "MMD"
    if 'DRCN' in l: return "DRCN"
    if 'baseline' in l: return "Baseline"

  all_runs = []
  for run in runs:
    experiment_type = check_exp(run.tags)
    if ((not direction) or  (run.tags and (direction in run.tags))):
      res_dict = {"name": run.name, 'experiment_type': experiment_type, 'id': run.id, "tags": run.tags}
      res_dict.update({k:v for k,v in run.summary.items() if 'gradient' not in k})
      res_dict.update(run.config)
      if 'best_accuracy' not in res_dict and "Best accuracy" in  run.config:
        res_dict['best_accuracy'] = run.config['Best accuracy']

      all_runs.append(res_dict)

  col_to_keep = ['id', 'name', 'experiment_type', 'best_test_loss', 'best_accuracy', 'tags']
  df = pd.DataFrame(all_runs).loc[:, col_to_keep].sort_values('best_accuracy', ascending=False)
  return df.reset_index(drop=True)


class LogSize(nn.Module):
  "utility class for control the size of the image trough the process"
  def __init__(
      self, 
      txt: str = '', 
      test: bool = True
    ) -> None:

    super().__init__()
    self.txt = txt
    self.test = test

  def forward(self, batch):
    if self.test:
      print(f"{self.txt}: {batch.size()}")
    return batch


class FlattenToChannelView(nn.Module):
  "utility class for control the size of the image trough the process"
  def __init__(
      self, 
      c: int, 
      h: int, 
      w: int
    ) -> None:

      super().__init__()
      self.c = c 
      self.h = h 
      self.w = w 

  def forward(self, x):
    return x.reshape(x.size(0), self.c, self.h, self.w)


class ChannelToFlatView(nn.Module):
  "utility class for flattening the output."
  def __init__(
      self, 
      dim_1: int = 256*5*5
    ) -> None:

      super().__init__()
      self.dim_1 = dim_1 

  def forward(self, x):
    return x.view(-1, self.dim_1)


def check_grad_names(model):
  print(model.name)
  for n, p in model.named_parameters():
    if p.requires_grad:
      print(f"Layer with gradients active:\t {n}")
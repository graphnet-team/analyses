import pickle
import numpy as np

import torch
from torch import Tensor

from graphnet.components.loss_functions import LossFunction

print('imported LogCoshLoss_Weighted')

with open('/home/iwsatlas1/timg/remote/icedata/convert/loss_weight.pkl', 'rb') as f:
    loss_weight = pickle.load(f)


class LogCoshLoss_Weighted(LossFunction):
    """Log-cosh loss function.

    Acts like x^2 for small x; and like |x| for large x.
    """
    @classmethod
    def _log_cosh(cls, x: Tensor) -> Tensor:  # pylint: disable=invalid-name
        """Numerically stable version on log(cosh(x)).

        Used to avoid `inf` for even moderately large differences.
        See [https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617]
        """
        return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implementation of loss calculation."""
        diff = prediction - target
        weight = loss_weight(target.cpu())
        elements = self._log_cosh(diff) * torch.Tensor(weight).to(diff.device)
        return elements

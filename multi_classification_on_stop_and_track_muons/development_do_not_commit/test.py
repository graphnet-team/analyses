from torch.nn.functional import one_hot
import torch

one_hot(torch.arange(0, 5) % 3, num_classes=5)

test = 2
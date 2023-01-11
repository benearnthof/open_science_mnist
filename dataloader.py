# this file contains both the train and test loader classes needed 
# to load the mnist dataset
from utils import Parameters
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# TODO: swap out torchvision transforms with data augmentation

params = Parameters()

# adapted from: https://github.com/pytorch/examples/blob/main/mnist/main.py

class TrainLoader(DataLoader):
  """Helper Class to simplify loading of training data."""
  def __init__(self):
    super().__init__(dataset=MNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                              ])), batch_size=params.train_batch_size, shuffle=True)


class TestLoader(DataLoader):
  """Helper Class to simplify loading of test data."""
  def __init__(self):
    super().__init__(dataset=MNIST('../data', train=False, download=True,
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                              ])), batch_size=params.test_batch_size, shuffle=True)

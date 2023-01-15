# this file contains both the train and test loader classes needed 
# to load the mnist dataset
from transforms import rotate_and_flip, normalize
from utils import Parameters
from dataset import AlbumentationsMnist
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

params = Parameters()

# adapted from: https://github.com/pytorch/examples/blob/main/mnist/main.py

class TrainLoader(DataLoader):
  """Helper Class to simplify loading of training data."""
  def __init__(self):
    super().__init__(dataset=AlbumentationsMnist(
                              root='../data',
                              train=True, 
                              download=True,
                              transform=rotate_and_flip
                              ), 
                            batch_size=params.train_batch_size, 
                            shuffle=True)

class TestLoader(DataLoader):
  """Helper Class to simplify loading of training data."""
  def __init__(self):
    super().__init__(dataset=AlbumentationsMnist(
                              root='../data',
                              train=False, 
                              download=True,
                              transform=normalize
                              ), 
                            batch_size=params.test_batch_size, 
                            shuffle=True)

class TrainLoaderNormalized(DataLoader):
  """Helper Class to simplify loading of training data."""
  def __init__(self):
    super().__init__(dataset=AlbumentationsMnist(
                              root='../data',
                              train=True, 
                              download=True,
                              transform=normalize
                              ), 
                            batch_size=params.train_batch_size, 
                            shuffle=True)

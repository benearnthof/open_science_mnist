# This file contains utility functions like a class to store data
# and stuff to visualize the dataset
import numpy as np
import matplotlib.pyplot as plt

class Parameters():
  """Helper Class to wrap up all training parameters"""
  def __init__(self, train_batch_size=32, test_batch_size=1000, epochs=10, lr=0.01,
               gamma=0.7, cuda=False, log_interval=100, save_model=False):
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.epochs = epochs
    self.lr = lr
    self.gamma = gamma
    self.cuda = cuda
    self.log_interval = log_interval
    self.save_model = save_model

def quick_plot(X, y):
  labs = y.tolist()
  """Helper function to visualize data and check if the data loaders worked on the fly"""
  for i, (img, y) in enumerate(zip(X[:5].reshape(5, 28, 28), labs[:5])):
    plt.subplot(151 + i)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(y)

# source https://github.com/iam-mhaseeb/Multi-Layer-Perceptron-MNIST-with-PyTorch/blob/master/mnist_mlp_exercise.ipynb
def plot_25(images, labels):
  """Helper function to plot a 5x5 grid of MNIST digits"""
  fig = plt.figure(figsize=(25,25))
  for idx in np.arange(25):
    ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]))
    ax.set_title(str(labels[idx].item()))

def plot_1(image):
  """Helper function to plot a single digit with its respective pixel values"""
  img = image
  fig = plt.figure(figsize=(25,25))
  ax = fig.add_subplot(111)
  ax.imshow(img)
  width, height = img.shape
  thresh = img.max()/2.5
  for x in range(width):
      for y in range(height):
          val = round(img[x][y],2) if img[x][y] !=0 else 0
          ax.annotate(str(val), xy=(y,x),
                      horizontalalignment='center',
                      verticalalignment='center',
                      color='white' if img[x][y]<thresh else 'black')

# we overwrite the `__getitem__` method of MNIST so we can inherit all the other 
# useful methods of this class & only have to redefine the transform
from torchvision.datasets import MNIST
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

class AlbumentationsMnist(MNIST):
  """Custom Class to extend MNIST to use Albumentation transforms."""
  def __init__(self, **kwds):
    super().__init__(**kwds)

  def __getitem__(self, index: int):
    img, target = self.data[index], int(self.targets[index])

    # this is required to keep consistent with MNIST
    img = Image.fromarray(img.numpy(), mode="L")
    # this is then required to use Albumentation transforms
    img = np.array(img)

    if self.transform is not None:
      augmented = self.transform(image=img)
      img = augmented['image']

    return img, target

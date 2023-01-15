# File for image transforms and augmentations we pass to our dataset
# To use custom augmentations add to these simply by using Albumentations.Compose
# https://albumentations.ai/docs/api_reference/core/composition/
import albumentations as A
from albumentations.pytorch import ToTensorV2

flip = A.Compose(
  [
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.1307,), std=(0.3081,)),
    ToTensorV2(),
  ]
)

rotate = A.Compose(
  [
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.1307,), std=(0.3081,)),
    ToTensorV2(),
  ]
)

rotate_and_flip = A.Compose(
  [
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.1307,), std=(0.3081,)),
    ToTensorV2(),
  ]
)

normalize = A.Compose(
  [
    A.Normalize(mean=(0.1307,), std=(0.3081,)),
    ToTensorV2(),
  ]
)

# script to evaluate the performance of our models on mnist
import torch
import torch.nn.functional as F
import numpy as np
from utils import Parameters

params = Parameters()


def test_model(model, test_loader):
  """A simple wrapper for model evaluation"""
  test_loss = 0.0
  loss_function = F.nll_loss
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  model.eval() # prep model for *evaluation*

  for data, target in test_loader:
      output = model(data)
      loss = loss_function(output, target)
      test_loss += loss.item()*data.size(0)
      # convert output logits to class predictions
      _, pred = torch.max(output, 1)
      correct = np.squeeze(pred.eq(target.data.view_as(pred)))
      # calculate test accuracy for each object class
      for i in range(params.test_batch_size):
          label = target.data[i]
          class_correct[label] += correct[i].item()
          class_total[label] += 1

  # calculate and print avg test loss
  test_loss = test_loss/len(test_loader.dataset)
  print('Test Loss: {:.6f}\n'.format(test_loss))

  for i in range(10):
      if class_total[i] > 0:
          print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
              str(i), 100 * class_correct[i] / class_total[i],
              np.sum(class_correct[i]), np.sum(class_total[i])))
      else:
          print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

  print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
      100. * np.sum(class_correct) / np.sum(class_total),
      np.sum(class_correct), np.sum(class_total)))

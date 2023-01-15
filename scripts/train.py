# simple script to train a model
import torch
import torch.nn.functional as F

def train_model(model, params, train_loader):
  model.train()
  # negative log likelihood loss 
  # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
  loss_function = F.nll_loss
  # optimizer: standard SGD
  optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
  for epoch in range(params.epochs):
    # monitor training loss
    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        # apply model to data & calculate loss
        output = model(data)
        loss = loss_function(output, target)
        # calculate gradients & perform optim.step
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    # print training statistics 
    # calculate average loss
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.4f}'.format(
        epoch+1, 
        train_loss
        ))
  # operates on model parameters in place returns nothing
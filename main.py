import torch 
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from IPython.display import clear_output


def train(model, 
          train_data, 
          optimizer,
          loss_fn, 
          mode='any', 
          val_data=None, 
          epochs=None,     
          batch_size=None 
         ):

  """
  Train a given model using the specified parameters and data.
    
  Parameters:
  - model (torch.nn.Module): The model to be trained, typically an instance from the 'encoder' folder.
  - train_data (torchvision.datasets.FashionMNIST() or torch.utils.data.DataLoader()): The training data. If using batched data, 
                                                                                       provide a DataLoader object; otherwise, a Dataset object can be supplied.
  - optimizer (torch.optim.Adam()): The optimization algorithm.
  - loss_fn: The loss function. For Autoencoders (AE), use nn.MSELoss(). 
             For Variational Autoencoders (VAE), this should be None, and vae_loss_fn from this module will be used instead.
  - mode: 
        'any' (default): Use for vanilla AE, multilayer AE, convolutional AE, or denoising AE.
        'sparse'       : Use for sparse AE, which includes additional regularization in the loss.
        'vae'          : Use for any variational AE, which will require a custom loss function.
  - val_data (torchvision.datasets.FashionMNIST() or torch.utils.data.DataLoader()): The validation data. If using batched data, 
                                                                                     provide a DataLoader object; otherwise, a Dataset object can be supplied
  - epochs (int): The number of training epochs, specifically for batched data. A typical value might be 20.
  - batch_size (int): The batch size to be used during training. Only specify this if your loss function requires input flattening, as in a vanilla VAE.

  Returns:
  total_loss: The history of losses while training.
  """
                   
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  total_loss = []
  model.train()

  if mode == 'vae':
   for epoch in range(epochs):
      epoch_loss = 0.0
     
      # training
      for i, batch in enumerate(train_data):
        optimizer.zero_grad()
        data = batch[0].type(torch.float).to(device)
        decoded_data, mu, log_var = model(data)
        
        if decoded_data.shape[1] == data.size(2) * data.size(3):
          loss = loss_fn(data, decoded_data, mu, log_var, flatten=True, batch_size=batch_size)
        else:
          loss = loss_fn(data, decoded_data, mu, log_var)
          
        epoch_loss += loss.item() 
        total_loss.append(loss.item())
        if (i % 500) == 0:
          drawing(data[0, 0, :, :].reshape(28, 28).cpu().detach().numpy(),
                  batch[1][0].item(),
                  decoded_data[0, 0, :, :].reshape(28, 28).cpu().detach().numpy(),
                  loss.item(),
                  wait=False)
        loss.backward()
        optimizer.step()
      print(f'Loss on train: \tepoch {epoch+1} / {epochs}: \tMSE_loss: {epoch_loss:.5f}')
     
      # validating
      mean_acc =  []
      with torch.no_grad():
        model.eval()
        for j, batch in enumerate(val_data):
          x, y = batch[0].type(torch.float).to(device), batch[1]
          decoded_data_val, mu_val, log_var_val = model(x)
          
          if decoded_data.shape[1] == data.size(2) * data.size(3):
            loss = loss_fn(data, decoded_data, mu, log_var, flatten=True, batch_size=batch_size)
          else:
            loss = loss_fn(data, decoded_data, mu, log_var)
        
          mean_acc.append(loss.item())
          if (j % 200) == 0:
            drawing(x[0, 0, :, :].reshape(28, 28).cpu().detach().numpy(),
                    y[0].item(),
                    decoded_data_val[0, 0, :, :].reshape(28, 28).cpu().detach().numpy(),
                    loss.item(),
                    wait=False)
      print(f'Loss on validation: {sum(mean_acc):.5f}\n')
     
  else:
    for x, y in train_data:
      optimizer.zero_grad()
      x = torch.squeeze(x).type(torch.float)
      output = model(x)
      loss = loss_fn(output.reshape(-1), x.reshape(-1))
      
      if mode == 'sparse':
        all_layer1_params = torch.cat([x.view(-1) for x in model.layer1.parameters()])
        all_layer2_params = torch.cat([x.view(-1) for x in model.layer2.parameters()])
        loss += torch.norm(all_layer1_params, 1) * model.coef_l1
        loss += torch.norm(all_layer2_params, 1) * model.coef_l1
  
      total_loss.append(loss.item())
      loss.backward()
      optimizer.step()
    
  return total_loss


def eval(model, validator, loss_fn):
  total = len(validator)
  iter, mean_acc = 0, []
  with torch.no_grad():
    model.eval()
    
    for x, y in validator:
      x = torch.squeeze(x).type(torch.float)
      output = model(x)
      loss = loss_fn(output.reshape(-1), x.reshape(-1))
      mean_acc.append(loss.item())

      if (iter + 1) % 500 == 0 or iter == 0:
        if cton[y] == 'Sneaker':
          drawing(x, y, output, loss.item())
      iter += 1

  return sum(mean_acc) / len(mean_acc)

# do a -- class to number -- dictionary for FashionMNIST dataset
cton = {i:item for i, item in enumerate(torchvision.datasets.FashionMNIST.classes)}


def drawing(x, y, output, acc):
  plt.suptitle(f'Class: {cton[y]}  \nMSE: {acc:.3f}')
  plt.subplot(1, 2, 1)
  plt.imshow(x.reshape(28, 28))
  plt.title('\nOriginal')

  plt.subplot(1, 2, 2)
  plt.imshow(output.reshape(28, 28))
  plt.title(f'Restored')
  plt.show()
  
  clear_output(wait=True)


def vae_loss_fn(x, decoded_data, mu, log_var, flatten=False, batch_size=None):
  if flatten:
    x = x.view(batch_size, -1)
    decoded_data = decoded_data.view(batch_size, -1)
  
  mse_loss = nn.MSELoss(reduction='sum')
  MSE = mse_loss(decoded_data, x)
  KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
  return MSE + KLD

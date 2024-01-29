import torch 
import torchvision
import matplotlib.pyplot as plt
from IPython.display import clear_output


def train(model, trainer, optimizer, loss_fn, mode='any', validator=None, epochs=None, ):
  """
  mode: 
        --- any ---- if vanilla AE, multilayer AE, convolutional AE, denoising AE (default)
        -- sparse -- if sparse AE (add regularization)
        --- vae ---- if any variational AE
  """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  total_loss = []
  model.train()

  if mode == 'vae':
   for epoch in range(epochs):
      epoch_loss = 0.0
     
      # training
      for i, batch in enumerate(trainer):
        optimizer.zero_grad()
        data = batch[0].type(torch.float).to(device)
        decoded_data, mu, log_var = model(data)
        loss = vae_loss_fn(data, decoded_data, mu, log_var)
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
        for j, batch in enumerate(validator):
          x, y = batch[0].type(torch.float).to(device), batch[1]
          decoded_data_val, mu_val, log_var_val = model(x)
          loss = vae_loss_fn(decoded_data_val, x, mu_val, log_var_val)
          mean_acc.append(loss.item())
          if (j % 200) == 0:
            drawing(x[0, 0, :, :].reshape(28, 28).cpu().detach().numpy(),
                    y[0].item(),
                    decoded_data_val[0, 0, :, :].reshape(28, 28).cpu().detach().numpy(),
                    loss.item(),
                    wait=False)
      print(f'Loss on validation: {sum(mean_acc):.5f}\n')
     
  else:
    for x, y in trainer:
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
  total = len(val)
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


def vae_loss_fn(x, decoded_data, mu, log_var):
  mse_loss = nn.MSELoss(reduction='sum')
  MSE = mse_loss(decoded_data, x)
  KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
  return MSE + KLD

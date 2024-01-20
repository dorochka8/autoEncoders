import torch 

def train(model, trainer, optimizer, loss_fn, mode='any', epochs=100):
  """
  mode: 
        --- any ---- if vanilla AE, multilayer AE, convolutional AE, denoising AE (default)
        -- sparse -- if sparse AE
        --- vae ---- if variational AE
  """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  total_loss = []
  model.train()

  if mode == 'vae':
    for epoch in range(epochs):
    epoch_loss, batches = 0.0, 0
    
    for i, batch in enumerate(trainer):
      batches += 1
      optimizer.zero_grad()
      data = batch[0].type(torch.float).reshape(batch[0].shape[0], -1)
      data = data.to(device)
      decoded_data, mu, log_var = model(data)
      loss = loss_fn(decoded_data, data, mu, log_var)
      epoch_loss += loss.item() / len(data)
      total_loss.append(loss.item() / len(data))

      loss.backward()
      optimizer.step()
    print(f'Loss on epoch {epoch+1} / {epochs}: \tMSE_loss: {epoch_loss:.3f}\n')

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

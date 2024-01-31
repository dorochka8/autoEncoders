import torch
import torch.nn as nn 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAE(nn.Module):
  def __init__(self, in_features, latent_space, batch_size):
    super().__init__()
    self.latent_space = latent_space
    self.batch_size   = batch_size
    
    self.encoder = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=500),
        nn.ReLU())

    self.mu      = nn.Linear(in_features=500, out_features=self.latent_space)
    self.log_var = nn.Linear(in_features=500, out_features=self.latent_space)

    self.decoder = nn.Sequential(
        nn.Linear(in_features=self.latent_space, out_features=500),
        nn.ReLU(),
        nn.Linear(in_features=500, out_features=in_features), 
        nn.Sigmoid())


  def reparametrization_trick(self, mu, log_var):
    std = torch.exp(log_var * 0.5)
    eps = torch.randn_like(std)
    return mu + eps * std


  def forward(self, x):
    x = x.view(self.batch_size, -1)

    # encodoer
    x = self.encoder(x)                      # batch_size x 500

    # mu & log_var
    mu = self.mu(x)
    log_var = self.log_var(x)
    z = self.reparametrization_trick(mu, log_var)
    
    # decoder
    decoded = self.decoder(z)                # batch_size x in_features 

    return decoded, mu, log_var


  def sampling(self, samples):
    with torch.no_grad():
      z = torch.randn(samples, self.latent_space).to(device)
      decoded = self.decoder(z)
    return decoded

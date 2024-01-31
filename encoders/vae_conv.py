import torch
import torch.nn as nn 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvVAE(nn.Module):
  def __init__(self, in_channels, latent_space, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.origin_shape = None
    self.idx1 = None
    self.idx2 = None

    self.encoder_pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
    self.encoder1 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3),
        nn.ReLU(),
    )
    self.encoder2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4),
        nn.ReLU(),
    )
    self.encoder3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2),
        nn.ReLU(),
    )
    self.bottleneck1 = nn.Linear(in_features=512, out_features=latent_space)

    self.mu      = nn.Linear(in_features=latent_space, out_features=latent_space)
    self.log_var = nn.Linear(in_features=latent_space, out_features=latent_space)

    self.bottleneck2 = nn.Linear(in_features=latent_space, out_features=512)
    self.decoder_unpool = nn.MaxUnpool2d(kernel_size=2)
    self.decoder1 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2),
        nn.ReLU(),
    )
    self.decoder2 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4),
        nn.ReLU(),
    )
    self.decoder3 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, kernel_size=3),
        nn.Sigmoid()
    )


  def reparametrization_trick(self, mu, log_var):
    std = torch.exp(log_var * 0.5)
    eps = torch.randn_like(std)
    return mu + eps * std


  def forward(self, x):
    # encodoer
    x = self.encoder1(x)                     # batch_size x 32 x 26 x 26
    x, idx1 = self.encoder_pool(x)           # batch_size x 32 x 13 x 13
    x = self.encoder2(x)                     # batch_size x 32 x 10 x 10
    x, idx2 = self.encoder_pool(x)           # batch_size x 32 x  5 x  5
    encoded = self.encoder3(x)               # batch_size x 32 x  4 x  4
    self.origin_shape = encoded.shape
    self.idx1 = idx1
    self.idx2 = idx2

    # flatten and reshape before mu & log_var
    x = encoded.view(self.batch_size, -1)         # batch_size x 512
    x = self.bottleneck1(x)                  # batch_size x latent_space
    
    # mu & log_var
    mu = self.mu(x)
    log_var = self.log_var(x)
    z = self.reparametrization_trick(mu, log_var)

    # reshape and unflatten after reparametrication
    x = self.bottleneck2(z)
    x = x.view(*self.origin_shape)            # batch_size x 32 x  4 x  4
    
    # decoder
    x = self.decoder1(x)                      # batch_size x 32 x  5 x  5
    x = self.decoder_unpool(x, self.idx2)     # batch_size x 32 x 10 x 10
    x = self.decoder2(x)                      # batch_size x 32 x 13 x 13
    x = self.decoder_unpool(x, self.idx1)     # batch_size x 32 x 26 x 26
    decoded = self.decoder3(x)                # batch_size x 1  x 28 x 28

    return decoded, mu, log_var


  def sampling(self, samples):
    with torch.no_grad():
      z = torch.randn(samples, *self.origin_shape[1:]).to(device)
      x = self.decoder1(z)
      x = self.decoder_unpool(x, self.idx2[:samples, :, :, :])
      x = self.decoder2(x)
      x = self.decoder_unpool(x, self.idx1[:samples, :, :, :])
      decoded = self.decoder3(x)
    return decoded

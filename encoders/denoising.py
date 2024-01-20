class DenoisingEncoder(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    # encoder
    self.enc_conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=3, padding='same')
    self.enc_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
    self.enc_pool  = nn.MaxPool2d(kernel_size=2)

    # decoder
    self.dec_conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
    self.dec_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
    self.dec_pool  = nn.Upsample(scale_factor=2)

    # output
    self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same')

  def forward(self, x):
    noise = torch.zeros(x.size(), dtype=torch.float32) * torch.randn(x.size())
    x = x + noise
    x = x.unsqueeze(0).unsqueeze(0)

    # encoder
    x = self.enc_pool(F.relu(self.enc_conv1(x)))
    x = self.enc_pool(F.relu(self.enc_conv2(x)))

    # decoder
    x = self.dec_pool(F.relu(self.dec_conv1(x)))
    x = self.dec_pool(F.relu(self.dec_conv2(x)))

    # output
    x = self.output(x)
    return x

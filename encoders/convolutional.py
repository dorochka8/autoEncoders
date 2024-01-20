class ConvAutoencoder(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    # encoder
    self.enc_conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=3, padding='same')
    self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same')
    self.enc_conv3 = nn.Conv2d(in_channels=8,  out_channels=8, kernel_size=3, padding='same')
    self.enc_pool  = nn.MaxPool2d(kernel_size=2)

    # decoder
    self.dec_conv1 = nn.Conv2d(in_channels=8, out_channels=8,  kernel_size=3, padding='same')
    self.dec_conv2 = nn.Conv2d(in_channels=8, out_channels=8,  kernel_size=3, padding='same')
    self.dec_conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
    self.dec_pool  = nn.Upsample(scale_factor=2)
    self.dec_pool_get_back_shape = nn.Upsample(scale_factor=2.5)

    # output
    self.output = nn.Conv2d(in_channels=16, out_channels=input_size, kernel_size=3, padding='same')

  def forward(self, x):
    x = x.unsqueeze(0).unsqueeze(0)

    # encoder
    x = self.enc_pool(F.relu(self.enc_conv1(x)))
    x = self.enc_pool(F.relu(self.enc_conv2(x)))
    x = self.enc_pool(F.relu(self.enc_conv3(x)))

    # decoder
    x = self.dec_pool_get_back_shape(F.relu(self.dec_conv1(x)))
    x = self.dec_pool(F.relu(self.dec_conv2(x)))
    x = self.dec_pool(F.relu(self.dec_conv3(x)))

    # output
    x = self.output(x)
    return x

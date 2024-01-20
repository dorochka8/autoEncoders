class MultilayerAutoencoder(nn.Module):
  def __init__(self, input_size, hidden_size, code_size):
    super().__init__()
    self.layer1 = nn.Linear(in_features=input_size,  out_features=hidden_size)
    self.layer2 = nn.Linear(in_features=hidden_size, out_features=code_size)
    self.layer3 = nn.Linear(in_features=code_size, out_features=input_size)

  def forward(self, x):
    x = x.reshape(-1)
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = self.layer3(x)
    return x

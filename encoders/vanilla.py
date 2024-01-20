class VanillaAutoencoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.layer1 = nn.Linear(in_features=input_size,  out_features=hidden_size)
    self.layer2 = nn.Linear(in_features=hidden_size, out_features=output_size)

  def forward(self, x):
    x = x.reshape(-1)
    x = F.relu(self.layer1(x))
    x = self.layer2(x)
    return (x)

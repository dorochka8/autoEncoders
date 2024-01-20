class SparseEncoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, coef_l1):
    super().__init__()
    self.layer1 = nn.Linear(in_features=input_size,  out_features=hidden_size)
    self.layer2 = nn.Linear(in_features=hidden_size, out_features=output_size)
    self.coef_l1 = coef_l1

  def forward(self, x):
    x = x.reshape(-1)
    x = F.relu(self.layer1(x))
    x = self.layer2(x)
    return x

  def l1(self, weight):
    return torch.sum(torch.abs(weight))

import torch

class Net(torch.nn.Module):   
  def __init__(self):
      super(Net, self).__init__()

      self.cnn_layers = torch.nn.Sequential(
          # Defining a 2D convolution layer
          torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
          # Defining another 2D convolution layer
          torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(inplace=True),
          torch.nn.MaxPool2d(kernel_size=2, stride=2),
      )

      self.linear_layers = torch.nn.Sequential(
          torch.nn.Linear(64 * 7 * 7, 63)
      )

  # Defining the forward pass    
  def forward(self, x):
      x = self.cnn_layers(x)
      x = x.view(x.size(0), -1)
      # print(x.size)
      x = self.linear_layers(x)
      return x
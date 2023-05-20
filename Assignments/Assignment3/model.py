# model.py
import torch
import torch.nn as nn

from torchmetrics.classification import Accuracy as Acc

# Device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Model Defination
class FedModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = 3, stride = 1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
    )
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = 128*16*16, out_features = 512),
        nn.ReLU(),
        nn.Linear(in_features = 512, out_features = 10)
    )

  def forward(self, x):
    x = self.conv_block(x)
    # print(x.shape)
    x = self.fc(x)

    return x
  

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Accuray Function
accuracy_fn = Acc("multiclass",num_classes = 10).to(device)


if __name__ == "__main__":
  model_test = FedModel().to(device)
  rand_input = torch.randn((1,3,32,32)).to(device)
  preds = model_test(rand_input)
  print(f"shape of preds: {preds.shape}")
# clientx.py
import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from tqdm.auto import tqdm



# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# print(parent)

from model import FedModel, accuracy_fn
from engine import train_client

# Device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Make dataset
client1_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=client1_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=client1_transform, download=True)

# Trime dataset
class_count = [0 for i in range(10)]
train_count = 0
i = 0
train_data = []
while train_count != 10000:
  image = train_dataset[i][0]
  label = train_dataset[i][1]

  if class_count[label] >= 1000:
    i += 1
    continue
  else:
    train_data.append((image, label))
    class_count[label] += 1
    train_count += 1
    i += 1
  # print(f"count: {train_count}")

class_count = [0 for i in range(10)]
test_count = 0
i = 0
test_data = []
while test_count != 5000:
  image = test_dataset[i][0]
  label = test_dataset[i][1]

  if class_count[label] >= 500:
    i += 1
    continue
  else:
    test_data.append((image, label))
    class_count[label] += 1
    test_count += 1
    i += 1
  # print(f"count: {test_count}")

client1_train_subset = torch.utils.data.ConcatDataset([train_data])
client1_test_subset = torch.utils.data.ConcatDataset([test_data])

BATCH_SIZE = 32
client1_train_dataloader = DataLoader(dataset = client1_train_subset, batch_size = BATCH_SIZE, shuffle = True)
client1_test_dataloader = DataLoader(dataset = client1_test_subset, batch_size = BATCH_SIZE, shuffle = False)

client_1_model = FedModel().to(device)
client_1_loss_fn = nn.CrossEntropyLoss()

def update_client(client1_train_dataloader, client1_test_dataloader, client_1_model, client_1_loss_fn, accuracy_fn, device):
    print("\n CLIENT 1:")
    

    # load model
    if os.path.exists('weights/client1.pth'):
        # print("\nload c1") 
        MODEL_NAME = "client1.pth"
        MODEL_PATH_SAVE = "weights/" + MODEL_NAME

        if torch.cuda.is_available() == False:
            client_1_model.load_state_dict(torch.load(f = MODEL_PATH_SAVE, map_location=torch.device('cpu')))
        else:
            client_1_model.load_state_dict(torch.load(f = MODEL_PATH_SAVE))

    client_1_optimizer = torch.optim.Adam(params = client_1_model.parameters(), lr = 1e-3)

    client_1_model, client_1_loss_fn, client_1_optimizer, test_acc = train_client(client_1_model, client1_train_dataloader, client1_test_dataloader,
                                                                        client_1_optimizer, client_1_loss_fn, accuracy_fn, device = device)


    # save model
    MODEL_PATH = Path("weights")
    MODEL_PATH.mkdir(parents = True, exist_ok = True)

    MODEL_NAME = "client1.pth"
    MODEL_PATH_SAVE = MODEL_PATH / MODEL_NAME

    print(f"model saved at: {MODEL_PATH_SAVE}")
    if os.path.exists('weights/client1.pth'):
        os.remove("weights/client1.pth")
        torch.save(obj = client_1_model.state_dict(), f = MODEL_PATH_SAVE)
    else:
        torch.save(obj = client_1_model.state_dict(), f = MODEL_PATH_SAVE)

    print()
    # Confusion Matrix
    y_preds = []

    with torch.inference_mode():
        for x, y in tqdm(client1_test_dataloader, desc = "Making prediction..."):
            x, y = x.to(device), y.to(device)

            logit = client_1_model(x)
            pred = torch.softmax(logit.squeeze(), dim = 1).argmax(dim = 1)

            y_preds.append(pred)

    y_tensor_preds = torch.cat(y_preds).to('cpu')



    target = [j for (i,j) in client1_test_subset]

    confmat = ConfusionMatrix(num_classes = 10, task = 'multiclass')
    confmat_tensor = confmat(preds = y_tensor_preds, target = torch.from_numpy(np.array(target)))

    fix, ax = plot_confusion_matrix(conf_mat = confmat_tensor.numpy(), figsize = (10,7))

    plt.show()

    print(f"\n Overall Accuracy of Client Model: {test_acc:.4f}")
    print("\n")
    # Classwise Accuracy
    classwise_acc = confmat_tensor.diag()/confmat_tensor.sum(1)

    for i in range(len(classwise_acc)):
        print(f"Class {i}: {classwise_acc[i]:.4f}")

    print("\n")
    return client_1_model, client_1_loss_fn, client_1_optimizer


def get_client1():
   global client1_train_dataloader, client1_test_dataloader, client_1_model, client_1_loss_fn, accuracy_fn, device

   client_1_model, client_1_loss_fn, client_1_optimizer = update_client(client1_train_dataloader, client1_test_dataloader, client_1_model, client_1_loss_fn, accuracy_fn, device)

   return  client_1_model, client_1_loss_fn, client_1_optimizer
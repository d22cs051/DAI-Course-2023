# engine.py
import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torch import nn
import torchmetrics

# Device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def training_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    acc_fn: torchmetrics.classification.accuracy.MulticlassAccuracy,
    optimizer: torch.optim.Optimizer,
    device: str,
    profiler: torch.profiler.profile = None,
):
    """
    Desc:
      funtion to perform traning step for one EPOCH
    Args:
      model (nn.Module): Pytorch model class object
      dataloader (torch.utils.data.DataLoader): training dataloder from training dataset
      loss_fn (nn.Module): Loss Function (object) of your choice
      acc_fn (torchmetrics.classification.accuracy.MulticlassAccuracy): accuracy function from trochmetrics
      optimizer (torch.optim.Optimizer): Optimizer Function (object) of your choice
      device (str): Torch Device "CPU/GPU"
      profiler (torch.profiler.profile, optional): Pytorch Profiler. Defaults to None.
    Returns:
      train_loss (float), train_acc (float): training loss and training accuracy for one EPOCH
    """
    model.train()  # putting model in traing model

    train_loss, train_acc = 0, 0  # initlizing loss and acc. for the epoch

    if profiler != None:
      profiler.start()
      for step,(X, y) in enumerate(dataloader):  # loop in batches
        if step >= (1 + 1 + 3) * 2:
          break
        X, y = X.to(device), y.to(device)  # sending the data to target device
        # print(f"shape of X: {X.shape}, shape of y: {y.shape}")
        
        # 1. forward pass
        y_pred_logits = model(X)
        # y_pred = y_pred_logits.argmax(dim=1).type(torch.int)
        # print(y_pred)
        # 2. calculate the loss
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()

        # 3. optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backword
        loss.backward()

        # 5. optimizer step
        optimizer.step()
        
        train_acc += acc_fn(y_pred_logits, y).item()
        
        profiler.step()
      profiler.stop()
    else:
      for step,(X, y) in enumerate(dataloader):  # loop in batches
        X, y = X.to(device), y.to(device)  # sending the data to target device
        # print(f"shape of X: {X.shape}, shape of y: {y.shape}")
        
        # 1. forward pass
        y_pred_logits = model(X)
        # y_pred = y_pred_logits.argmax(dim=1).type(torch.int)
        # print(y_pred)
        # 2. calculate the loss
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()

        # 3. optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backword
        loss.backward()

        # 5. optimizer step
        optimizer.step()
        
        train_acc += acc_fn(y_pred_logits, y).item()
    # 6. returning actual loss and acc.x
    return train_loss / len(dataloader), train_acc / len(dataloader), model




def testing_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    acc_fn: torchmetrics.classification.accuracy.MulticlassAccuracy,
    device: str,
):
    """
    Desc:
      funtion to perform testing step for one EPOCH
    Args:
      model (nn.Module): Pytorch model class object
      dataloader (torch.utils.data.DataLoader): testing dataloder from training dataset
      loss_fn (nn.Module): Loss Function (object) of your choice
      acc_fn (torchmetrics.classification.accuracy.MulticlassAccuracy): accuracy function from trochmetrics
      device (str): Torch Device "CPU/GPU"
    Returns:
      test_loss (float), test_acc (float): testing loss and testing accuracy for one EPOCH
    """
    model.eval()  # putting model in eval model

    test_loss, test_acc = 0, 0  # initlizing loss and acc. for the epoch

    # with torch.inference_mode(): # disabling inference mode for aqcuiring gradients of perturbed data
    for (X, y) in dataloader:  # loop in batches
        X, y = X.to(device), y.to(device)  # sending the data to target device
        # print(f"shape of X: {X.shape}, shape of y: {y.shape}")

        # 1. forward pass
        y_pred_logits = model(X)

        # 2. calculate the loss
        loss = loss_fn(y_pred_logits, y)
        test_loss += loss.item()

        # printing the prediction and actual label
        # print(y_pred_logits.argmax(dim=1), y,sep='\n')
        
        # 3. calculating accuracy
        test_acc += acc_fn(y_pred_logits, y).item()

    # 6. returning actual loss and acc.
    return test_loss / len(dataloader), test_acc / len(dataloader)
  
  

def eval_func(
  model: torch.nn.Module,
  dataloader: torch.utils.data.DataLoader,
  loss_fn: torch.nn.Module, accuracy_fn, 
  device: str
  ):
  
  eval_loss, eval_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for x_eval, y_eval in dataloader:
      
      if device == 'cuda':
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

      # 1. Forward
      eval_pred = model(x_eval)
      
      # 2. Loss and accuray
      eval_loss += loss_fn(eval_pred, y_eval)
      eval_acc += accuracy_fn(y_eval, torch.argmax(eval_pred, dim=1))


    eval_loss /= len(dataloader)
    eval_acc /= len(dataloader)

  # print(eval_loss, eval_acc)
  return eval_loss, eval_acc

# train client
def train_client(client_model, train_dataloader, test_dataloader, client_optimizer, client_loss_fn, accuracy_fn, device = device):
  # init. epochs
  epoches = 10

  client_model_train_loss, client_model_test_loss = [], []
  client_model_train_accs, client_model_test_accs = [], []
  print()

  start_time = timer()
  torch.manual_seed(64)
  torch.cuda.manual_seed(64)
  for epoch in tqdm(range(epoches)):
    # print(f"Epoch: {epoch+1}")
    train_loss, train_acc, client_model = training_step(model = client_model, dataloader = train_dataloader,
                                      loss_fn = client_loss_fn, optimizer = client_optimizer,
                                      accuracy_fn = accuracy_fn, device = device)
    
    test_loss, test_acc = testing_step(model = client_model, dataloader = test_dataloader,
                                    loss_fn = client_loss_fn, accuracy_fn = accuracy_fn,
                                    device = device)
    
    client_model_train_loss.append(train_loss.item())
    client_model_test_loss.append(test_loss.item())
    client_model_train_accs.append(train_acc.item())
    client_model_test_accs.append(test_acc.item())


    # print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
    # print()

  # plot_graph(model_18_train_loss, model_18_test_loss, model_18_train_accs, model_18_test_accs)

  end_time = timer()

  return client_model, client_optimizer, client_loss_fn, test_acc
  # print(f"Execution time: {end_time - start_time} Seconds.")

  # save_model('cifair100_model_18.pth', model_18)
  # print("Model saved")
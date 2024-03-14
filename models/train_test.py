import torch
from tqdm import tqdm


def train_step(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str
    ) -> tuple[float]:
    """Train step of a model

    :param model: Model object
    :param dataloader: Dataloader containing training image
    :param criterion: Loss function
    :param optimizer: Optimizer object
    :param device: Device, normally cuda or cpu
    :return: Training loss and accuracy of step
    """
    model.train()
    train_loss, train_acc = 0, 0
    
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward Propagation
        y_pred = model(X)
        
        # Loss calc
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Backwards propagation
        loss.backward()
        
        # Update values
        optimizer.step()
        
        # Calculate accuracy
        pred_label = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (pred_label == y).sum().item()/len(y_pred)
    
    train_loss, train_acc = train_loss/len(dataloader), train_acc/len(dataloader)
    
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    device: str
    ) -> tuple[float]:
    """Test step of a model

    :param model: Model object
    :param dataloader: Dataloader containing test image
    :param criterion: Loss function
    :param device: Device, normally cuda or cpu
    :return: Test loss and accuracy of step
    """
    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Forward Propagation
            y_pred = model(X)
            
            # Loss calc
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            
            # Calculate accuracy
            pred_label = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (pred_label == y).sum().item()/len(y_pred)
            
    test_loss, test_acc = test_loss/len(dataloader), test_acc/len(dataloader)
    
    return test_loss, test_acc


def train(
    model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader, 
    test_dataloader: torch.utils.data.DataLoader, 
    epochs: int,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
    ) -> dict[str, list[float]]:
    """Full training of model

    :param model: Model object
    :param train_dataloader: Dataloader containing training image
    :param test_dataloader: Dataloader containing test image
    :param epochs: Amount of epochs to train
    :param criterion: Loss function
    :param optimizer: Optimizer object
    :param device: Device, normally cuda or cpu
    :return: Metrics: loss and accuracy
    """
    metrics = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
    }
    for epoch in range(1, epochs+1):
        print("EPOCH", epoch)
        train_loss, train_acc = train_step(model, train_dataloader, criterion, optimizer, device)
        test_loss, test_acc = test_step(model,test_dataloader, criterion, device)
        metrics
        print(f"Epoch {epoch} - {train_loss=:.4f}, {train_acc=:.4f}, {test_loss=:.4f}, {test_acc=:.4f}")
    return metrics

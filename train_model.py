from models.data_setup import generate_dataloaders
from models.model import (
    load_pre_trained_model,
    prepare_for_transfer_learn,
    save_model
)
from models.train_test import train

from torch.nn import CrossEntropyLoss
from torch.cuda import is_available
from torch.optim import Adam


def main():
    device = "cuda" if is_available() else "cpu"
    batch_size = 4
    print(device)
    
    train_dl, test_dl, _ = generate_dataloaders("data", batch_size)
    model = load_pre_trained_model().to(device)
    prepare_for_transfer_learn(model)
    loss_fn = CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=1e-3)
    train(model, train_dl, test_dl, 5, loss_fn, optim, device)
    save_model(model, "models\model.pth")

if __name__ == "__main__":
    main()

import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(232, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def generate_dataloaders(path: str, batch_size: int) -> tuple[DataLoader, DataLoader, list[str]]:
    """Generates training and testing dataloaders from a dataset in given path
    The dataset must be arranged as such:
    .../class1/x1.png
    .../class1/x2.png
    .../class1/x3.png
    .../class2/x1.png
    .../class2/x2.png
    .../class2/x3.png

    :param path: Path to the dataset
    :param batch_size: Batch size of data loader
    :return: Train and test dataloader and class names
    """
    
    data = datasets.ImageFolder(path, transform)
    test_data, train_data = random_split(data, [0.3, 0.7])
    
    workers = max(os.cpu_count() - 1, 1)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=workers)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=workers)
    
    return train_loader, test_loader, data.classes

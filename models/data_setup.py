import os
import pandas as pd
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset


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

    :param path: Path to the datasets folder
    :param batch_size: Batch size of data loader
    :return: Train and test dataloader and class names
    """
    animal_data = datasets.ImageFolder(os.path.join(path, "animals"), transform)
    
    # Sampling randomly to get same amount of bg as other classes
    num_samples = round(len(animal_data)/len(animal_data.classes))
    df_ids = pd.read_csv(os.path.join(path, "metadata.csv"))["image_id"].sample(num_samples)
    bg_data = NotAnimalDataset(os.path.join(path, "images"), df_ids, transform)
    
    data = ConcatDataset([animal_data, bg_data])
    test_data, train_data = random_split(data, [0.2, 0.8])
    
    workers = max(os.cpu_count() - 1, 1)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=workers)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=workers)
    
    return train_loader, test_loader, animal_data.classes + ["nenhum"]


class NotAnimalDataset(Dataset):
    """Dataset with images without animals in them
    """
    CLASS_LABEL = 90
    
    def __init__(self, path: str, file_names: pd.DataFrame, transform: transforms.Compose = None) -> None:
        super().__init__()
        self.file_names = file_names
        self.path = path
        self.transform = transform
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image_name = f"{self.file_names.iloc[index]:0>7}"
        image_path = os.path.join(self.path, f"{image_name}.jpg")
        
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        
        return image, self.CLASS_LABEL

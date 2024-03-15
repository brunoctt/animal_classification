import json
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

try:
    from models.data_setup import transform
except ModuleNotFoundError:
    from data_setup import transform


def load_pre_trained_model() -> nn.Module:
    """Loads ResNet50 with DEFAULT weights (Imagenet1kV2)

    :return: Torch model with pre-trained weights
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    return model


def prepare_for_transfer_learn(model: nn.Module, output_size: int = 91, device="cuda"):
    """Freezes model's layers and changes the amount of output features of the fully connected layer
    to be the amount of classes in the dataset.
    A dropout layer is also added to avoid overfitting

    :param model: Model object
    :param output_size: Amount of classes in dataset, defaults to 90
    """
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=model.fc.in_features, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=output_size),
    ).to(device)


def classify_image(model: nn.Module, transform: transforms.Compose, image_path: str, device: str = "cuda") -> int:
    """Classifies an image from a given path

    :param model: Model object
    :param transform: Transforms to be applied to image
    :param image_path: Path to load image
    :param device: Device, normally cuda or cpu
    :return: Prediction label
    """
    image = Image.open(image_path)
    
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        image_pred = model(image)
    
    pred_label = torch.argmax(torch.softmax(image_pred, dim=1), dim=1)
    return pred_label[0].item()


def save_model(model: nn.Module, path: str = 'model.pth'):
    """Saves model state dict (weights)

    :param model: Model object
    :param path: Path to save file, defaults to 'model.pth'
    """
    torch.save(model.state_dict(), path)


def load_model(path: str = 'model.pth') -> nn.Module:
    """Loads model with saved state

    :param path: Path to saved state file, defaults to 'model.pth'
    :return: Model object
    """
    loaded_model = load_pre_trained_model()
    prepare_for_transfer_learn(loaded_model)
    loaded_model.load_state_dict(torch.load(f=path))
    return loaded_model


def get_label_names() -> dict[int, str]:
    """Loads json with label names and converts number string into ints

    :return: Dict with label names
    """
    with open("classes.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    return {int(i): name for i, name in classes.items()}


if __name__ == "__main__":
    model = load_model()
    pred = classify_image(model, transform, r"..\test_folder\dog.jpeg") 
    print(get_label_names()[pred])

import os
from torch.cuda import is_available

from models.data_setup import transform
from interface.select_folder import Interface
from models.model import load_model, classify_image, get_label_names


def main():
    gui = Interface()
    gui.open_selector()
    
    device = "cuda" if is_available() else "cpu"
    labels = get_label_names()
    
    model = load_model("models\model.pth").to(device)
    folder_path = gui.folder_path
    
    for file_name in os.listdir(folder_path):
        try:
            file_path = os.path.join(folder_path, file_name)
            res = classify_image(model, transform, file_path, device)
            print(file_name, "-", labels[res])
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    main()

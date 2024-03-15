import os
import pandas as pd
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
    
    df = {"imagens": [], "classificacao": []}
    
    for file_name in os.listdir(folder_path):
        try:
            file_path = os.path.join(folder_path, file_name)
            res = classify_image(model, transform, file_path, device)
            df["imagens"].append(file_name)
            df["classificacao"].append(labels[res])
        except Exception as e:
            print(e)
            continue
    
    df = pd.DataFrame.from_dict(df)
    df.to_csv("results.csv", index=False)
    print("Classificação finalizada")


if __name__ == "__main__":
    main()

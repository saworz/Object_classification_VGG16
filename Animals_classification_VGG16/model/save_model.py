import torchvision
import torch
import sys
from pathlib import Path


def save_model(model: torchvision.models, classes: str) -> None:

    save_path = input("Input path to save model: ")

    if save_path == "quit":
        sys.exit()
    
    Path(save_path + "\Save").mkdir(parents=True, exist_ok=True)

    with open(Path(save_path) / "Save" / "model.pt","w") as f:
        torch.save(model.state_dict(), Path(save_path) / "Save" / "model.pt")

    with open(Path(save_path) / "Save" / "classes.txt", "w") as f:
        f.write('\n'.join(classes))

    print("Model saved.")

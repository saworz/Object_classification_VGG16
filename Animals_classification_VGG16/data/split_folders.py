import splitfolders
import os
from pathlib import Path

def split_folders(images_path: Path) -> None:

    
    input_folder = images_path

    if os.path.isdir(input_folder / "train"):
        print("Data already splitted.")
        return
    else:
        print("Splitting files into train and validation.")

        splitfolders.ratio(
            input_folder,
            output=input_folder,
            seed=42,
            ratio=(0.8, 0.2),
            move=True,
            group_prefix=None
        )
        print("Moving files finished.")

        os.listdir(input_folder)

        for i in os.listdir(input_folder):
            if len(os.listdir(input_folder/i)) == 0:
                os.rmdir(input_folder/i)
        print("Empty folders deleted.")
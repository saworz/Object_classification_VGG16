from pathlib import Path
import os
from zipfile import ZipFile

def extract_zip(path: str) -> None:

    data_path = Path(path)
    image_path = data_path.parent / "Dataset"

    if image_path.is_dir():
        print(f"{image_path} exists.")
    else:
        print(f"{image_path} doesn't exist, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    if not os.listdir(image_path):
        with ZipFile(data_path, 'r') as zip:
            print("Extracting files...")
            zip.extractall(image_path)
            
            print("Extracting finished.")
    else:
        print("Data already extracted.")
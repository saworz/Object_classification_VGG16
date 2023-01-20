import torchvision
from pathlib import Path
from PIL import Image
from typing import Tuple
import torch
from torch.utils.data import Dataset

class ImageFolderCustom(Dataset):

    def __init__(self, targ_dir: str, transform: torchvision.transforms) -> None:

        self.paths = list(Path(targ_dir).glob("*.jpg"))
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        return self.transform(img)
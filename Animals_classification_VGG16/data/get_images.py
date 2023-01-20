import os
from os import listdir

def get_images(folder_dir: str) -> list:

    images = []
    for file in os.listdir(folder_dir):

        if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
            images.append(file)

    return images
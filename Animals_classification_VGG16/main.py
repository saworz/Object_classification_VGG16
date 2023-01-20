from features.image_folder import ImageFolderCustom
from features.predict_custom_image import predict_custom_images
from features.handle_input import handle_input
from model.save_model import save_model
from data.get_images import get_images

from pathlib import Path
import torchvision
import torch
import sys

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Working on a device: {device}")
    print("""--type 'quit' to exit--""")
    model, classes = handle_input(device)

    while True:

        print("--Input 'quit' to leave--")
        custom_data_dir = input("""Enter image path or type 'save' to save model: """)

        if custom_data_dir == 'quit':
            sys.exit()

        try:
            custom_data_transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.ToTensor()]
            )

            if custom_data_dir == 'save':
                save_model(model, classes)
            else:
                print("\nProcessing...")
                custom_data = ImageFolderCustom(
                    targ_dir=custom_data_dir,
                    transform=custom_data_transform
                )

                images = get_images(custom_data_dir)

                predict_custom_images(
                    model=model,
                    data=custom_data,
                    classes=classes,
                    images=images,
                    device=device
                )
        except :
            print("Something went wrong, try again...")

if __name__ == '__main__':
    main()
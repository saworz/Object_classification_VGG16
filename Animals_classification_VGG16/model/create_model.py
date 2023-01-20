from data.extract_dataset import extract_zip
from data.split_folders import split_folders
from data.parse_data import train_options
from model.training import train

import sys
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from typing import Tuple

def create_model(device: torch.device) -> tuple([torchvision.models, list]):

    opt = train_options()

    while True:
        
        print("Required path to the dataset. Dataset has to be in .zip format.")
        dataset_path = input('Enter dataset path:')
        
        if dataset_path == 'quit':
            sys.exit()

        try:
            extract_zip(dataset_path)
            split_folders(Path(dataset_path).parent / "Dataset")

            train_transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.ToTensor()]
            )

            val_transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.ToTensor()]
            )

            train_dir = Path(dataset_path).parent / "Dataset" / "train"
            val_dir = Path(dataset_path).parent / "Dataset" / "val"

            train_data = torchvision.datasets.ImageFolder(
                root=train_dir,
                transform=train_transform
            )

            val_data = torchvision.datasets.ImageFolder(
                root=val_dir,
                transform=val_transform
            )
            break
        except:
            print("Something went wrong, try again...")

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=opt.workers
    )

    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=opt.batch,
        shuffle=False,
        num_workers=opt.workers
    )

    image_batch, label_batch = next(iter(train_dataloader))

    weights = torchvision.models.VGG16_Weights.DEFAULT
    auto_transforms = weights.transforms()

    model = torchvision.models.vgg16(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    output_shape = len(train_data.classes)

    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([torch.nn.Linear(num_features, len(train_data.classes))])  
    model.classifier = torch.nn.Sequential(*features)
    model.to(device)
    
    torch.cuda.manual_seed(opt.seed)
    torch.manual_seed(opt.seed)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=opt.lr
    )

    start_time = timer()

    model_results, preds = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=opt.epochs,
        device=device
    )

    end_time = timer()
    print(f"Total learning time: {(end_time - start_time):.3f}")

    return model, train_data.classes
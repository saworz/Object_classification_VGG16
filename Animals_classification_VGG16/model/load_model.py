import sys
import torch
import torchvision
from pathlib import Path
from typing import Tuple

def load_model() -> tuple([torchvision.models, list]):

    base_model = torchvision.models.vgg16()

    while True:
        print("Required path for saved trained model's weights")
        load_path = input("Input path: ")

        if load_path == 'quit':
            sys.exit()

        try:
            loaded_model = torch.load(load_path)
            break
        except:
            print("Incorrect path.")
    
    classifier = loaded_model.popitem(last=True)
    classifier_size = len(classifier[1])
    
    num_features = base_model.classifier[6].in_features
    features = list(base_model.classifier.children())[:-1]
    features.extend([torch.nn.Linear(num_features, classifier_size)])  
    base_model.classifier = torch.nn.Sequential(*features)

    base_model.load_state_dict(torch.load(load_path))
    
    with open(Path(load_path).parent / "classes.txt") as file:
        classes = [line.rstrip() for line in file]

    return base_model, classes
    
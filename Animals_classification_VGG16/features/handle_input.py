from model.create_model import create_model
from model.load_model import load_model

import torchvision
import torch
import sys
from typing import Tuple

def handle_input(device: torch.device) -> tuple([torchvision.models, list]):
    print("Do you want to train a new model or load existing weights?")

    while(True):
        print("Enter T for training or L for loading")
        chosen_option = input()

        if chosen_option == 'T' or chosen_option == 't':
            model, classes = create_model(device)
            print("Model trained succesfully.")
            return model, classes
        elif chosen_option == 'L' or chosen_option == 'l':
            model, classes = load_model()
            model.to(device)
            print("Model loaded succesfully.")
            return model, classes
        elif chosen_option == 'quit':
            sys.exit()
        else:
            print('Incorrect input')
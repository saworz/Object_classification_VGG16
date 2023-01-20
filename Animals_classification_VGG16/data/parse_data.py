import argparse

def train_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=8, type=int, help='batch size')
    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--epochs", default=1, type=int, help='amount of epochs')
    parser.add_argument("--workers", default=4, type=int,help='number of workers')
    parser.add_argument("--seed", default=42, type=int,help='random seed')
    opt = parser.parse_args()
    return opt
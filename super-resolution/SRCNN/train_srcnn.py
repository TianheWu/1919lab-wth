import argparse


def get_parse():
    parser = argparse.ArgumentParser(description='Train SRCNN')

    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()
    return args


def train(epochs, net, train_data):
    net.train()
    for epoch in range(epochs):
        pass
    


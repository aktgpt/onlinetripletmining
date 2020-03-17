import argparse
import json
import warnings

from numba.errors import NumbaPerformanceWarning
from torchvision import datasets, transforms

from networks import ConvNet
from train import TripletTrainer

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


def main(config):
    train_dataset = datasets.MNIST(
        root="", train=True, download=True, transform=transforms.ToTensor(),
    )
    test_dataset = datasets.MNIST(
        root="", train=False, download=True, transform=transforms.ToTensor()
    )

    model = ConvNet()
    model = model.cuda()
    trainer = TripletTrainer(config)
    trainer.train(train_dataset, test_dataset, model)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train triplet dataset")
    argparser.add_argument("-c", "--conf", help="path to configuration file")
    args = argparser.parse_args(["-c", "config.json"])
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    main(config)

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from CNN import CNN
from Classifier import Classifier
from SlcDataset import SlcDataset
from KfoldDataLoader import KfoldDataLoader

parser = argparse.ArgumentParser(description='CNN model for classifying segmentation data')
parser.add_argument('manifest_path', help='Path to manifest file')
parser.add_argument('project_name', help='Project name')
parser.add_argument('--debug', default=False, required=False, action='store_true',
                    help='Whether to debug on a tiny sample')
args = parser.parse_args()

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)


def main():
    slcDataset = SlcDataset(manifest_path=args.manifest_path, project_name=args.project_name,
                            image_dim=(128, 128), debug=args.debug)

    if args.debug:
        train_loader = DataLoader(
            slcDataset,
            batch_size=64,
            shuffle=True)
        kfoldDataLoader = None
        test_loader = None
    else:
        train_dataset, train_y, test_dataset = slcDataset.get_train_test_datasets(test_size=.3, seed=1234)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64)
        kfoldDataLoader = KfoldDataLoader(train_dataset=train_dataset, y=train_y, n_splits=5, batch_size=64,
                                          shuffle=True, random_state=1234)

    model = CNN(input_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    classifier = Classifier(
        model=model,
        n_epochs=20,
        train_loader=train_loader,
        kfoldDataLoader=kfoldDataLoader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        save_path='./saved_models',
        debug=args.debug
    )
    classifier.fit()
    # classifier.evaluate()



if __name__ == '__main__':
    main()


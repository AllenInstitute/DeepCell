import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from CNN import CNN
from Classifier import Classifier
from SlcDataset import SlcDataset
from KfoldDataLoader import KfoldDataLoader
from Subset import Subset

parser = argparse.ArgumentParser(description='CNN model for classifying segmentation data')
parser.add_argument('manifest_path', help='Path to manifest file')
parser.add_argument('project_name', help='Project name')
parser.add_argument('model_config_path', help='Path to pickled model config')
parser.add_argument('-additional_train_transform_path', help='Data augmentation for train set', required=False)
parser.add_argument('-n_epochs', help='Number of training epochs', default=20, type=int)
parser.add_argument('--debug', default=False, required=False, action='store_true',
                    help='Whether to debug on a tiny sample')
args = parser.parse_args()

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

# Make results deterministic
torch.manual_seed(1234)


def main():
    model_config = torch.load(args.model_config_path)

    if args.additional_train_transform_path:
        additional_train_transform = torch.load(args.additional_train_transform_path)
    else:
        additional_train_transform = None

    slcDataset = SlcDataset(manifest_path=args.manifest_path, project_name=args.project_name,
                            image_dim=(128, 128), debug=args.debug)

    if args.debug:
        train_loader = DataLoader(
            Subset(dataset=slcDataset, indices=range(len(slcDataset)), apply_transform=True),
            batch_size=64,
            shuffle=True)
        kfoldDataLoader = None
        test_loader = None
    else:
        train_dataset, train_y, test_dataset = slcDataset.get_train_test_datasets(test_size=.3, seed=1234)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64)
        kfoldDataLoader = KfoldDataLoader(train_dataset=train_dataset, y=train_y, n_splits=5, batch_size=64,
                                          shuffle=True, random_state=1234,
                                          additional_train_transform=additional_train_transform)

    model = CNN(cfg=model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    classifier = Classifier(
        model=model,
        n_epochs=args.n_epochs,
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


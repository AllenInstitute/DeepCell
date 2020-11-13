import argparse

import torch

from CNN import CNN
from Classifier import Classifier
from SlcDataset import SlcDataset
from data_loaders import get_data_loaders

parser = argparse.ArgumentParser(description='CNN model for classifying segmentation data')
parser.add_argument('manifest_path', help='Path to manifest file')
parser.add_argument('project_name', help='Project name')
parser.add_argument('--debug', default=False, required=False, action='store_true',
                    help='Whether to debug on a tiny sample')
args = parser.parse_args()


def main():
    slcDataset = SlcDataset(manifest_path=args.manifest_path, project_name=args.project_name,
                            image_dim=(128, 128), debug=args.debug)

    train_split = not args.debug
    train_loader, test_loader = get_data_loaders(
        full_dataset=slcDataset, y=slcDataset.y, test_size=.3, batch_size=64, seed=1234, train_split=train_split
    )

    model = CNN(input_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    classifier = Classifier(
        model=model,
        n_epochs=20,
        train_loader=train_loader,
        valid_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        save_path='./saved_models'
    )
    # classifier.fit()
    classifier.evaluate()



if __name__ == '__main__':
    main()


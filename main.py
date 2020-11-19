import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from CNN import CNN
from Classifier import Classifier
from Plotting import Plotting
from SlcDataset import SlcDataset
from KfoldDataLoader import KfoldDataLoader
from Subset import Subset

manifest_path = 's3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020/20201020135214/expert_output/ophys-experts-slc-oct-2020/manifests/output/output.manifest'
project_name = 'ophys-experts-slc-oct-2020'

parser = argparse.ArgumentParser(description='CNN model for classifying segmentation data')
parser.add_argument('-model_config_path', help='Path to pickled model config')
parser.add_argument('-experiment_name', help='Experiment name')
parser.add_argument('-additional_train_transform_path', help='Data augmentation for train set', required=False)
parser.add_argument('-n_epochs', help='Number of training epochs', default=20, type=int)
parser.add_argument('-learning_rate', help='Learning rate', default=1e-3, type=float)
parser.add_argument('-dropout_prob', help='Dropout prob', default=0.5, type=float)
parser.add_argument('-weight_decay', help='Weight decay (L2 regularizaion)', default=0.0, type=float)
parser.add_argument('--crop_to_center', help='Whether to crop the input to the area surrounding the mask',
                    default=False, action='store_true')
parser.add_argument('--use_learning_rate_scheduler', help='Use learning rate scheduler', default=False,
                    action='store_true')
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

    slcDataset = SlcDataset(manifest_path=manifest_path, project_name=project_name,
                            image_dim=(128, 128), debug=args.debug)

    if args.debug:
        train_loader = DataLoader(
            Subset(dataset=slcDataset, indices=range(len(slcDataset)), apply_transform=True,
                   center_crop=args.crop_to_center, additional_transform=additional_train_transform),
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
                                          additional_train_transform=additional_train_transform,
                                          crop_to_center=args.crop_to_center)

    model = CNN(cfg=model_config, dropout_prob=args.dropout_prob)
    optimizer = lambda: torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50)
    criterion = torch.nn.BCEWithLogitsLoss()
    classifier = Classifier(
        model=model,
        n_epochs=args.n_epochs,
        train_loader=train_loader,
        kfoldDataLoader=kfoldDataLoader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        save_path='./saved_models',
        use_learning_rate_scheduler=args.use_learning_rate_scheduler,
        debug=args.debug
    )
    cv_res = classifier.cross_validate()
    plotting = Plotting(experiment_name=args.experiment_name)
    plotting.plot_train_loss(train_loss=cv_res['train_losses'])
    plotting.plot_train_val_F1(train_f1=cv_res['all_train_f1'], val_f1=cv_res['all_val_f1'])
    # classifier.evaluate()



if __name__ == '__main__':
    main()


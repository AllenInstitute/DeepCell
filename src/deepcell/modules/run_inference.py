import json
from pathlib import Path

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from imgaug import augmenters as iaa

from deepcell.datasets.model_input import ModelInput
from deepcell.inference import inference
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.transform import Transform
from deepcell.models.VggBackbone import VggBackbone


def run_inference_for_experiment(
        experiment_id,
        rois_path: Path,
        data_dir: Path,
        model_weights_path: Path,
        output_path: Path):
    """
    Runs inference for experiment and produces csv of predictions

    Args:
        experiment_id:
            Experiment id to run predictions on
        rois_path:
            Path to ROIs. Needs to be in json format and each record needs
            to have key "id"
        data_dir:
            Path to artifacts for inference
        model_weights_path:
            Path to Pytorch model weights that were saved using torch.save
        output_path:
            Path to save predictions csv
        use_cuda:
            True if on GPU
    Returns:
        None, but writes predictions csv to disk
    """
    all_transform = transforms.Compose([
        iaa.Sequential([
            iaa.CenterCropToFixedSize(height=60, width=60)
        ]).augment_image,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = Transform(all_transform=all_transform)

    with open(rois_path) as f:
        rois = json.load(f)
        roi_ids = [f'{experiment_id}_{x["id"]}' for x in rois]

    model_inputs = [
        ModelInput.from_data_dir(data_dir=data_dir,
                                 experiment_id=experiment_id,
                                 roi_id=roi_id) for roi_id in roi_ids]
    test = RoiDataset(dataset=model_inputs,
                      transform=test_transform)
    test_dataloader = DataLoader(dataset=test, shuffle=False, batch_size=64)

    cnn = torchvision.models.vgg11_bn(pretrained=True, progress=False)
    cnn = VggBackbone(model=cnn, truncate_to_layer=15,
                      classifier_cfg=[512, 512], dropout_prob=.7,
                      freeze_up_to_layer=15)
    _, inference_res = inference(model=cnn, test_loader=test_dataloader,
                                 has_labels=False,
                                 checkpoint_path=str(model_weights_path))
    inference_res['experiment_id'] = experiment_id

    inference_res.to_csv(output_path / f'{experiment_id}_inference.csv',
                         index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', required=True,
                        help='What experiment to run inference on')
    parser.add_argument('--rois_path', required=True, help='Path to rois')
    parser.add_argument('--data_dir', required=True, help='Path to artifacts')
    parser.add_argument('--model_weights_path', required=True,
                        help='Path to trained model weights')
    parser.add_argument('--out_path', required=True, help='Where to store '
                                                          'predictions')
    args = parser.parse_args()

    rois_path = Path(args.rois_path)
    data_dir = Path(args.data_dir)
    model_weights_path = Path(args.model_weights_path)
    out_path = Path(args.out_path)

    if args.use_cuda not in('true', 'false'):
        raise ValueError('use_cuda must be one of "true" or "false"')

    use_cuda = args.use_cuda == 'true'

    run_inference_for_experiment(experiment_id=args.experiment_id, rois_path=rois_path,
                                 data_dir=data_dir, output_path=out_path,
                                 model_weights_path=model_weights_path)

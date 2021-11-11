import json
from pathlib import Path

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from imgaug import augmenters as iaa

from Inference import inference
from RoiDataset import RoiDataset
from Transform import Transform
from models.VggBackbone import VggBackbone


def main(experiment_id, rois_path: Path, data_dir: Path,
         model_weights_path: Path, output_path: Path, use_cuda=True):
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
        roi_ids = [x['id'] for x in rois]

    test = RoiDataset(manifest_path=None, data_dir=data_dir,
                      roi_ids=roi_ids,
                      transform=test_transform, has_labels=False,
                      parse_from_manifest=False)
    test_dataloader = DataLoader(dataset=test, shuffle=False, batch_size=64)

    cnn = torchvision.models.vgg11_bn(pretrained=True, progress=False)
    cnn = VggBackbone(model=cnn, truncate_to_layer=15,
                      classifier_cfg=[512, 512], dropout_prob=.7,
                      freeze_up_to_layer=15)
    _, inference_res = inference(model=cnn, test_loader=test_dataloader,
                                 has_labels=False,
                                 checkpoint_path=str(model_weights_path),
                                 use_cuda=use_cuda)
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
    parser.add_argument('--use_cuda', default=False, help='Whether on GPU')

    args = parser.parse_args()

    rois_path = Path(args.rois_path)
    data_dir = Path(args.data_dir)
    model_weights_path = Path(args.model_weights_path)
    out_path = Path(args.out_path)

    main(experiment_id=args.experiment_id, rois_path=rois_path,
         data_dir=data_dir, output_path=out_path, use_cuda=args.use_cuda,
         model_weights_path=model_weights_path)

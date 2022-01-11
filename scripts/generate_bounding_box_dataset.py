"""Generates initial dataset for labeling bounding boxes to be used in the
VIA app"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from deepcell.datasets.visual_behavior_extended_dataset import \
    VisualBehaviorExtendedDataset


def get_bounding_box(contours):
    """Gets a bounding box that contains all contours"""
    boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        width = x + w
        height = y + h
        boxes.append([x, y, width, height])

    boxes = np.array(boxes)
    x1, y1 = np.min(boxes, axis=0)[:2]
    x2, y2 = np.max(boxes, axis=0)[2:]
    return (x1, y1), (x2, y2)


def main(artifact_path: Path, out_dir: Path):
    res = []
    dataset = VisualBehaviorExtendedDataset(
        artifact_destination=Path(artifact_path),
        exclude_projects=[
            'ophys-experts-slc-oct-2020_ophys-experts-go-big-or-go-home',
            'ophys-expert-danielsf-additions'])

    for x in dataset.dataset:
        with open(x.mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = np.array(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        (x1, y1), (x2, y2) = get_bounding_box(contours=contours)
        if x.correlation_projection_path is None:
            continue
        res.append({
            'filename': x.correlation_projection_path,
            'file_size': 1,
            'file_attributes': {},
            'region_count': 1,
            'region_id': 0,
            'region_shape_attributes': json.dumps({
                'name': 'rect',
                'x': int(x1),
                'y': int(y1),
                'width': int(x2 - x1),
                'height': int(y2 - y1)
            }),
            'region_attributes': {}
        })

    df = pd.DataFrame(res)
    df.to_csv(str(out_dir / 'bounding_boxes.csv'), index=False)


if __name__ == '__main__':
    main(artifact_path=Path('/tmp/artifacts'), out_dir=Path('~/'))



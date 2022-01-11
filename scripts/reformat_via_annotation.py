"""Reformats VIA annotation output into a more usable format"""
import json
import re
from pathlib import Path
import argparse


def reformat(input_path: Path, output_path: Path):
    """
    Reformats VIA annotations into a more friendly json format

    Args:
        input_path:
            Path to VIA annotations
        output_path:
            Path to output
    Returns:
        None. Writes file to disk
    """

    with open(input_path) as f:
        annotations = json.load(f)

    res = {}
    for k, v in annotations.items():
        exp_id, roi_id = re.findall(r'\d+', k)[:-1]
        bounding_box = v['regions'][0]['shape_attributes']
        bounding_box = {
            'x': bounding_box['x'],
            'y': bounding_box['y'],
            'width': bounding_box['width'],
            'height': bounding_box['height']
        }
        res[f'exp_{exp_id}_roi_{roi_id}'] = bounding_box

    with open(output_path / 'bounding_boxes.json', 'w') as f:
        f.write(json.dumps(res))


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_path', help='Path to VIA annotations',
                            required=True)
        parser.add_argument('--output_path', help='Output path',
                            required=True)
        args = parser.parse_args()

        input_path = Path(args.input_path)
        output_path = Path(args.output_path)

        reformat(input_path=input_path, output_path=output_path)
    main()
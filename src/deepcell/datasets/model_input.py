import json
import os
import shutil
from pathlib import Path
from typing import Optional, Union, List, Dict

from deepcell.datasets.channel import Channel, channel_filename_prefix_map


class ModelInput:
    def __init__(self,
                 roi_id: str,
                 experiment_id: str,
                 channel_path_map: Dict[Channel, Path],
                 channel_order: List[Channel],
                 project_name: Optional[str] = None,
                 label: Optional[str] = None):
        """
        A container for a single example given as input to the model

        Args:
            roi_id:
                ROI id
            experiment_id:
                Experiment id
            channel_path_map
                Map between `deepcell.roi_dataset.Channel` to Path
            project_name:
                optional name using to indicate a unique labeling job
                will be None if at test time (not labeled)
            label:
                optional label assigned to this example
                will be None if at test time (not labeled)
        """
        self._validate_channel_order(
            channel_order=channel_order,
            channel_path_map=channel_path_map
        )
        self._roi_id = roi_id
        self._experiment_id = experiment_id
        self._channel_path_map = channel_path_map
        self._channel_order = channel_order
        self._project_name = project_name
        self._label = label

    @property
    def roi_id(self) -> str:
        return self._roi_id

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def channel_path_map(self) -> Dict[Channel, Path]:
        return self._channel_path_map

    @property
    def channel_order(self) -> List[Channel]:
        return self._channel_order

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def project_name(self) -> str:
        return self._project_name

    @classmethod
    def from_data_dir(
            cls,
            data_dir: Union[str, Path],
            experiment_id: str,
            roi_id: str,
            channels: List[Channel],
            label: Optional[str] = None,
    ):
        """Instantiate a ModelInput from a data_dir.

        Args:
            data_dir:
                Path containing model inputs
            experiment_id
                Experiment id
            roi_id
                ROI id
            channels
                What channels to use
            label
                Label of roi either "cell" or "not cell".
        """
        data_dir = Path(data_dir)

        channel_filename_map = {
            c: f'{channel_filename_prefix_map[c]}_'
               f'{experiment_id}_{roi_id}.png'
            for c in channels
        }

        channel_path_map = {}
        for channel in channels:
            path = data_dir / channel_filename_map[channel]
            if not path.exists():
                raise ValueError(f'Expected channel {channel} to exist at '
                                 f'{path} but it did not')
            channel_path_map[channel] = path

        return ModelInput(
            experiment_id=experiment_id,
            channel_path_map=channel_path_map,
            roi_id=roi_id,
            label=label,
            channel_order=channels
        )

    def copy(self, destination: Path) -> None:
        """
        Copies to destination

        Parameters
        ----------
        destination: where to copy

        Returns
        -------
        None

        """
        for _, path in self.channel_path_map.items():
            shutil.copy(path, destination)

    def to_dict(self) -> dict:
        # deserialize from Channel to str so it can be json.dump
        channel_order = [x.value for x in self._channel_order]
        channel_path_map = {
            c.value: str(p) for c, p in self._channel_path_map.items()
        }

        return {
            'experiment_id': self._experiment_id,
            'roi_id': self._roi_id,
            'channel_order': channel_order,
            'channel_path_map': channel_path_map,
            'label': self._label,
            'project_name': self._project_name
        }

    @staticmethod
    def _validate_channel_order(
            channel_order: List[Channel],
            channel_path_map: Dict[Channel, Path]):

        for channel in channel_order:
            if channel not in channel_path_map:
                raise ValueError(
                    f'All channels in channel_order need to be in '
                    f'channel_path_map. {channel} not found in '
                    f'{channel_path_map}')


def write_model_input_metadata_to_disk(model_inputs: List[ModelInput],
                                       path: Union[str, Path]) -> None:
    """
    Writes a list of ModelInput to disk
    @param model_inputs: list of model inputs
    @param path: Where to write
    @return: None
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'w') as f:
        f.write(json.dumps([x.to_dict() for x in model_inputs], indent=2))


def copy_model_inputs_to_dir(
        destination: Path,
        model_inputs: List[ModelInput]) -> None:
    """Copies model inputs and metadata to `destination`
    @param destination: Destination
    @param model_inputs: List[ModelInput]
    """
    os.makedirs(destination, exist_ok=True)

    # Copy model inputs
    for model_input in model_inputs:
        model_input.copy(destination=destination)

    # Copy metadata
    write_model_input_metadata_to_disk(
        model_inputs=model_inputs,
        path=destination / 'model_inputs.json')

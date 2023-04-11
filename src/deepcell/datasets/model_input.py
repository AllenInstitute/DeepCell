import json
import os
import shutil
from pathlib import Path
from typing import Optional, Union, List


class ModelInput:
    def __init__(self,
                 roi_id: str,
                 experiment_id: str,
                 path: Path,
                 project_name: Optional[str] = None,
                 label: Optional[str] = None):
        """
        A container for a single example given as input to the model

        Args:
            roi_id:
                ROI id
            experiment_id:
                Experiment id
            path
                Path to input for this ROI
            project_name:
                optional name using to indicate a unique labeling job
                will be None if at test time (not labeled)
            label:
                optional label assigned to this example
                will be None if at test time (not labeled)
        """
        self._roi_id = roi_id
        self._experiment_id = experiment_id
        self._path = path
        self._project_name = project_name
        self._label = label

    @property
    def roi_id(self) -> str:
        return self._roi_id

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def path(self) -> Path:
        return self._path

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
            label
                Label of roi either "cell" or "not cell".
        """
        data_dir = Path(data_dir)

        path = data_dir / f'{experiment_id}_{roi_id}.npy'
        if not path.exists():
            raise ValueError(f'{path} does not exist')

        return ModelInput(
            experiment_id=experiment_id,
            roi_id=roi_id,
            label=label,
            path=path
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
        shutil.copy(self._path, destination)

    def to_dict(self) -> dict:
        # deserialize from Channel to str so it can be json.dump

        return {
            'experiment_id': self._experiment_id,
            'roi_id': self._roi_id,
            'path': str(self._path),
            'label': self._label
        }


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

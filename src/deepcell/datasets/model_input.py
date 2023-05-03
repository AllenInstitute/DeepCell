import json
import os
import shutil
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
from ophys_etl.types import OphysROI
from pydantic import BaseModel, Field


class _Peak(BaseModel):
    peak: int = Field(description='Index representing a peak in the trace')
    trace: float = Field(description='Value of the trace at peak')


class ModelInput:
    def __init__(self,
                 roi: OphysROI,
                 experiment_id: str,
                 ophys_movie_path: Path,
                 peaks: Optional[List[Dict]] = None,
                 peak: Optional[int] = None,
                 project_name: Optional[str] = None,
                 label: Optional[str] = None):
        """
        A container for a single example given as input to the model

        Args:
            roi:
                `OphysROI`
            experiment_id:
                Experiment id
            ophys_movie_path
                Path to ophys movie for this ROI
            peaks
                List of peak activation indices for this ROI
                Calculated using 
                ophys_etl.modules.roi_cell_classifier.compute_classifier_artifacts
                
                Either this or peak should be provided
            peak
                Use this specific peak. Either this or peaks should be provided
            project_name:
                optional name using to indicate a unique labeling job
                will be None if at test time (not labeled)
            label:
                optional label assigned to this example
                will be None if at test time (not labeled)
        """ # noqa E402
        if peaks is not None and peak is not None:
            raise ValueError('Provide peak or peaks, not both')
        if peaks is None and peak is None:
            raise ValueError('Provide peak or peaks, neither provided')

        self._roi = roi
        self._experiment_id = experiment_id
        self._ophys_movie_path = ophys_movie_path
        self._peaks = [_Peak(**x) for x in peaks]
        self._peak = peak
        self._project_name = project_name
        self._label = label

    @property
    def roi(self) -> OphysROI:
        return self._roi

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def ophys_movie_path(self) -> Path:
        return self._ophys_movie_path

    @property
    def peaks(self) -> Optional[List[_Peak]]:
        return self._peaks

    @property
    def peak(self) -> Optional[int]:
        return self._peak

    @peaks.setter
    def peaks(self, value):
        self._peaks = value

    @peak.setter
    def peak(self, value):
        self._peak = value

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

        path = data_dir / f'{experiment_id}' / f'{experiment_id}_{roi_id}.npy'
        if not path.exists():
            raise ValueError(f'{path} does not exist')

        return ModelInput(
            experiment_id=experiment_id,
            roi_id=roi_id,
            label=label,
            ophys_movie_path=path
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
        shutil.copy(self._ophys_movie_path, destination)

    def to_dict(self) -> dict:
        # deserialize from Channel to str so it can be json.dump

        return {
            'experiment_id': self._experiment_id,
            'roi': self._roi,
            'ophys_movie_path': str(self._ophys_movie_path),
            'label': self._label
        }

    def get_n_highest_peaks(self, n: int):
        """Gets the n highest peaks by trace value"""
        peak_indxs_sorted = np.argsort([-x.trace for x in self.peaks])
        return [self.peaks[i] for i in peak_indxs_sorted[:n]]


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

from pathlib import Path
from typing import Optional, Union


class ModelInput:
    def __init__(self,
                 roi_id: str,
                 experiment_id: str,
                 max_projection_path: Path,
                 avg_projection_path: Path,
                 mask_path: Path,
                 project_name: Optional[str] = None,
                 label: Optional[str] = None):
        """
        A container for a single example given as input to the model

        Args:
            roi_id:
                ROI id
            experiment_id:
                Experiment id
            max_projection_path:
                max projection path
            avg_projection_path:
                average projection path
            mask_path:
                mask path
            project_name:
                optional name using to indicate a unique labeling job
                will be None if at test time (not labeled)
            label:
                optional label assigned to this example
                will be None if at test time (not labeled)
        """
        self._roi_id = roi_id
        self._experiment_id = experiment_id
        self._max_projection_path = max_projection_path
        self._avg_projection_path = avg_projection_path
        self._mask_path = mask_path
        self._project_name = project_name
        self._label = label

    @property
    def roi_id(self) -> str:
        return self._roi_id

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def max_projection_path(self) -> Path:
        return self._max_projection_path

    @property
    def avg_projection_path(self) -> Path:
        return self._avg_projection_path

    @property
    def mask_path(self) -> Path:
        return self._mask_path

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def project_name(self) -> str:
        return self._project_name

    @classmethod
    def from_data_dir(cls, data_dir: Union[str, Path], experiment_id: str,
                      roi_id: str):
        """Instantiate a ModelInput from a data_dir
        This should be used at test time to construct inputs from a data_dir
        There are no labels in this scenario

        Args:
            data_dir:
                Path containing model inputs
            experiment_id
                Experiment id
            roi_id
                ROI id
        """
        data_dir = Path(data_dir)
        return ModelInput(
            experiment_id=experiment_id,
            avg_projection_path=data_dir / f'avg_{experiment_id}_{roi_id}.png',
            mask_path=data_dir / f'mask_{experiment_id}_{roi_id}.png',
            max_projection_path=data_dir / f'max_{experiment_id}_{roi_id}.png',
            roi_id=roi_id
        )
from pathlib import Path
from typing import Optional, Union, Dict


class ModelInput:
    def __init__(self,
                 roi_id: str,
                 experiment_id: str,
                 mask_path: Path,
                 max_projection_path: Path,
                 avg_projection_path: Optional[Path] = None,
                 correlation_projection_path: Optional[Path] = None,
                 project_name: Optional[str] = None,
                 label: Optional[str] = None,
                 bounding_box: Optional[Dict] = None):
        """
        A container for a single example given as input to the model

        Args:
            roi_id:
                ROI id
            experiment_id:
                Experiment id
            max_projection_path:
                max projection path
            correlation_projection_path:
                correlation projection path
            avg_projection_path:
                average projection path
            mask_path:
                mask path
            project_name:
                optional name using to indicate a unique labeling job
                will be None if at test time (not labeled)
            label:
                optional classification label assigned to this example
                will be None if at test time (not labeled)
            bounding_box
                A bounding box around the soma if present or around the
                mask if not present. Will be None at test time
        """
        if avg_projection_path is None and correlation_projection_path is None:
            raise ValueError('Must supply one of avg_projection_path or '
                             'correlation_projection_path')
        self._roi_id = roi_id
        self._experiment_id = experiment_id
        self._max_projection_path = max_projection_path
        self._correlation_projection_path = correlation_projection_path
        self._avg_projection_path = avg_projection_path
        self._mask_path = mask_path
        self._project_name = project_name
        self._label = label
        self._bounding_box = bounding_box

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
    def correlation_projection_path(self) -> Optional[Path]:
        return self._correlation_projection_path

    @property
    def avg_projection_path(self) -> Optional[Path]:
        return self._avg_projection_path

    @property
    def mask_path(self) -> Path:
        return self._mask_path

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def bounding_box(self) -> Optional[Dict]:
        return self._bounding_box

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
        def get_path(data_dir: Path, artifact_type: str, experiment_id: str,
                     roi_id: str):
            """Checks whether path format is
            <artifact_type>_<exp_id>_<roi_id> or
            <artifact_type>_exp_<exp_id>_roi_<roi_id>"""
            if (data_dir / f'{artifact_type}_{experiment_id}_'
                           f'{roi_id}.png').exists():
                return (data_dir / f'{artifact_type}_{experiment_id}_'
                           f'{roi_id}.png')
            elif (data_dir / f'{artifact_type}_exp_{experiment_id}_roi_'
                             f'{roi_id}.png').exists():
                return (data_dir / f'{artifact_type}_exp_{experiment_id}_roi_'
                             f'{roi_id}.png')
            else:
                raise RuntimeError(f'{artifact_type} could not be found for '
                                   f'experiment id {experiment_id}, roi id '
                                   f'{roi_id}')
        data_dir = Path(data_dir)

        try:
            correlation_projection_path = get_path(data_dir=data_dir,
                                                   artifact_type='corr',
                                                   experiment_id=experiment_id,
                                                   roi_id=roi_id)
        except RuntimeError:
            # correlation projection doesn't exist
            correlation_projection_path = None

        avg_proj_path = get_path(data_dir=data_dir,
                                 artifact_type='avg',
                                 experiment_id=experiment_id,
                                 roi_id=roi_id)
        mask_path = get_path(data_dir=data_dir,
                             artifact_type='mask',
                             experiment_id=experiment_id,
                             roi_id=roi_id)
        max_path = get_path(data_dir=data_dir,
                            artifact_type='max',
                            experiment_id=experiment_id,
                            roi_id=roi_id)

        return ModelInput(
            experiment_id=experiment_id,
            avg_projection_path=avg_proj_path,
            mask_path=mask_path,
            correlation_projection_path=correlation_projection_path,
            max_projection_path=max_path,
            roi_id=roi_id
        )

from pathlib import Path
from typing import Optional


class Artifact:
    def __init__(self,
                 roi_id: str,
                 experiment_id: str,
                 max_projection_path: Path,
                 avg_projection_path: Path,
                 mask_path: Path,
                 project_name: str,
                 label: Optional[str]):
        """
        Represents a single artifact given as input to the model

        Args:
            roi_id:
            experiment_id:
            max_projection_path:
            avg_projection_path:
            mask_path:
            project_name:
            label:
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

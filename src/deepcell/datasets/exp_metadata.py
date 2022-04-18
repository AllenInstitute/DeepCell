import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class ExperimentMetadata(TypedDict):
    """
    """
    experiment_id: str
    imaging_depth: int
    equipment: str
    problem_experiment: bool

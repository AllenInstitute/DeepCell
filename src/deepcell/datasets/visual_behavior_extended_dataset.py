import datetime
from pathlib import Path
from typing import Optional, List, Generator, Dict

from deepcell.datasets.visual_behavior_dataset import VisualBehaviorDataset


class VisualBehaviorExtendedDataset(VisualBehaviorDataset):
    def __init__(self, artifact_destination: Path,
                 exclude_projects: Optional[List[str]] = None,
                 download=True,
                 debug=False):
        """Represents the visual behavior dataset extended with additional
        data"""
        super().__init__(artifact_destination=artifact_destination,
                         exclude_projects=exclude_projects,
                         s3_manifest_prefix='visual_behavior_extended'
                                            '/manifests/',
                         download=download,
                         debug=debug)

    def _get_manifests(self) -> Dict[str, Generator[Dict, None, None]]:
        manifests = super()._get_manifests()

        def update_vasculature_manifest(manifest):
            new_manifest = []
            for x in manifest:
                roi_id = x['roi-id']
                exp_id = x['experiment-id']

                x[name] = {
                    # These ROIs are all not cell
                    'majorityLabel': 'not cell'
                }

                x[f'{name}-metadata'] = {
                    'creation-date': datetime.datetime(year=2021,
                                                       month=11,
                                                       day=11)
                }
                new_manifest.append(x)

            for x in new_manifest:
                yield x

        for i, (file, manifest) in enumerate(manifests.items()):
            file_name = Path(file).name
            if file_name == 'vasculature.manifest':
                name = 'add_vasculature'
                manifest = update_vasculature_manifest(manifest=manifest)
                manifests[file] = manifest
        return manifests

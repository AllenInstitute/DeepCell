import json
import subprocess
import sys
import time
from collections import defaultdict
import jsonlines
import os
import pandas as pd
import numpy as np

import visual_behavior.data_access.loading as loading
import slapp.utils.query_utils as qu

# NOTE requires inference-plots env (dependency conflicts)

def get_experiment_meta():
    creds = qu.get_db_credentials(env_prefix="LIMS_", **qu.lims_defaults)
    conn = qu.DbConnection(**creds)
    q = f'''
    SELECT movie_frame_rate_hz, e.name AS rig, oe.id AS experiment_id
    FROM ophys_experiments oe
    JOIN ophys_sessions os ON oe.ophys_session_id = os.id
    JOIN equipment e ON os.equipment_id = e.id
    JOIN specimens sp ON os.specimen_id = sp.id
    JOIN donors d ON sp.donor_id = d.id
    JOIN imaging_depths i ON i.id=oe.imaging_depth_id
    JOIN structures st ON st.id=oe.targeted_structure_id
    '''
    res = conn.query(q)
    df = pd.DataFrame(res)
    df = df.set_index('experiment_id')
    return df

def main():
    exp_to_local_to_global_map = defaultdict(dict)

    home_dir = os.path.expanduser('~')

    with open(f'{home_dir}/master_global_to_local_mapping.json') as f:
        global_to_local_mapping = json.loads(f.read())

    with open(f'{home_dir}/new') as f:
        new = json.loads(f.read())

    num_rois = 0

    with jsonlines.open(f'{home_dir}/output.manifest') as reader:
        for roi in reader:
            exp_id = roi['experiment-id']
            global_roi_id = int(roi['roi-id'])
            if global_roi_id not in new:
                continue
            local_roi_id = int(global_to_local_mapping['global_to_local'][str(global_roi_id)]['roi_id'])
            exp_to_local_to_global_map[exp_id][local_roi_id] = global_roi_id
            num_rois += 1
    num_exps = len(exp_to_local_to_global_map)
    print(f'num rois: {num_rois}')
    exp_meta = get_experiment_meta()

    for i, exp_id in enumerate(exp_to_local_to_global_map):
        start = time.time()

        print(f'EXP {i+1}/{num_exps}')

        movie_path = loading.get_motion_corrected_movie_h5_location(exp_id)
        local_to_global_roi_id_map = exp_to_local_to_global_map[exp_id]
        input_fps = exp_meta.loc[exp_id]['movie_frame_rate_hz']
        if np.isnan(input_fps):
            input_fps = 11.0 if exp_meta[exp_id]['rig'].startswith('MESO') else 31.0

        manifest = {
            'experiment_id': exp_id,
            'movie_path': movie_path,
            'binarized_rois_path': f'/allen/aibs/informatics/danielk/dev_LIMS/new_labeling/{exp_id}/binarize_output.json',
            'traces_h5_path': f'/allen/aibs/informatics/danielk/dev_LIMS/new_labeling/{exp_id}/roi_traces.h5',
            'local_to_global_roi_id_map': local_to_global_roi_id_map
        }
        input = {
          "prod_segmentation_run_manifest": f"{home_dir}/slapp_tform_manifest.json",
          "input_fps": input_fps,
          "log_level": "INFO",
          "artifact_basedir": "/allen/aibs/informatics/aamster/artifacts",
          "output_manifest": "/allen/aibs/informatics/aamster/artifacts/manifest.jsonl",
          "is_classifier_input": True,
          "cropped_shape": [64, 64]
        }

        with open(f'{home_dir}/slapp_tform_manifest.json', 'w') as f:
            f.write(json.dumps(manifest))

        with open(f'{home_dir}/slapp_tform_input.json', 'w') as f:
            f.write(json.dumps(input))

        cmd = f'python segmentation-labeling-app/slapp/transforms/transform_pipeline.py --input_json {home_dir}/slapp_tform_input.json'.split()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f'STDOUT: {result.stdout}')
        print(f'STDERR: {result.stderr}')

        end = time.time()
        duration = (end - start) / 60
        print(f'DONE in {duration:.0f} mins')

        sys.stdout.flush()


if __name__ == '__main__':
    main()
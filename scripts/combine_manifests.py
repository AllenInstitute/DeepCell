import json

from croissant.utils import read_jsonlines


def combine_and_save_manifests(manifests_metas, out='merged.manifest'):
    """

    :param manifests_metas: [{'manifest_url': ..., 'project_name': ...}]
    :param roi_id_less_than: includes only roi-ids less than this
    :return: None
    """
    combined = []

    new_project_name = 'ophys_experts'

    for manifest_meta in manifests_metas:
        manifest = read_jsonlines(manifest_meta['manifest_url'])
        project_name = manifest_meta['project_name']

        cells = 0.0
        noncells = 0.0

        invalid = 0.0

        for obs in manifest:
            if project_name in obs:
                if 'majorityLabel' in obs[project_name]:
                    if obs[project_name]['majorityLabel'] == 'cell':
                        cells += 1
                    else:
                        noncells += 1
                    obs[new_project_name] = obs[project_name]
                    del obs[project_name]
                    obs = json.dumps(obs)
                    combined.append(obs + '\n')
            else:
                invalid += 1
        print(project_name)
        print(f'cells: {cells}')
        print(f'noncells: {noncells}')
        print(f'invalid: {invalid}')

    with open(out, 'w') as f:
        f.writelines(combined)


if __name__ == '__main__':
    manifest_metas = [
        {
            'manifest_url': 's3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020/20201020135214/expert_output/ophys-experts-slc-oct-2020/manifests/output/output.manifest',
            'project_name': 'ophys-experts-slc-oct-2020'
            },
        {
            'manifest_url': 's3://prod.slapp.alleninstitute.org/behavior_3cre_1600roi/20201119175205/expert-output/ophys-experts-go-big-or-go-home-chain/manifests/output/output.manifest',
            'project_name': 'ophys-experts-go-big-or-go-home'
        },
        {
            'manifest_url': 's3://prod.slapp.alleninstitute.org/last-ditch-labeling-effort-20201222/combined_output.manifest',
            'project_name': 'ophys-expert-danielsf-additions'
        }
    ]
    combine_and_save_manifests(manifests_metas=manifest_metas, out='/tmp/output.manifest')

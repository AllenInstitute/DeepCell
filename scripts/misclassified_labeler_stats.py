from collections import defaultdict

import pandas as pd
from croissant.utils import read_jsonlines

project_name = 'ophys-experts-slc-oct-2020_ophys-experts-go-big-or-go-home'
manifest_path = 's3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020_behavior_3cre_1600roi_merged/output.manifest'


def calc_disagreement(manifest, roi_ids):
    roi_ids = set(roi_ids)
    manifest = [obs for obs in manifest if obs['roi-id'] in roi_ids]
    disagree = 0.0
    labeler_stats = defaultdict(lambda: defaultdict(int))

    for obs in manifest:
        if project_name in obs:
            if 'majorityLabel' in obs[project_name]:
                majority_label = obs[project_name]['majorityLabel']

                for annotation in obs[project_name]['workerAnnotations']:
                    labeler_stats[annotation['workerId']]['total'] += 1

                    if annotation['roiLabel'] != majority_label:
                        disagree += 1
                        labeler_stats[annotation['workerId']]['disagree'] += 1
                        if annotation['roiLabel'] == 'cell':
                            labeler_stats[annotation['workerId']]['cell_disagree'] += 1
                        else:
                            labeler_stats[annotation['workerId']]['noncell_disagree'] += 1

    disagree /= len(roi_ids)

    return disagree, labeler_stats


def combine_labeler_stats(stats_total, stats_misclassified):
    stats_total = pd.DataFrame(stats_total).T
    stats_total['frac_disagree'] = stats_total['disagree'] / stats_total['total']
    stats_total['frac_cell_disagree'] = stats_total['cell_disagree'] / stats_total['disagree']
    stats_total['frac_noncell_disagree'] = stats_total['noncell_disagree'] / stats_total['disagree']

    stats_misclassified = pd.DataFrame(stats_misclassified).T
    stats_misclassified['frac_disagree'] = stats_misclassified['disagree'] / stats_misclassified['total']
    stats_misclassified['frac_cell_disagree'] = stats_misclassified['cell_disagree'] / stats_misclassified['disagree']
    stats_misclassified['frac_noncell_disagree'] = stats_misclassified['noncell_disagree'] / stats_misclassified['disagree']

    df = stats_total.merge(stats_misclassified, left_index=True, right_index=True, suffixes=('_total', '_misclassified'))

    return df[['frac_disagree_total', 'frac_disagree_misclassified', 'frac_cell_disagree_total',
               'frac_cell_disagree_misclassified', 'frac_noncell_disagree_total',
               'frac_noncell_disagree_misclassified']]



def main():
    misclassified = pd.read_csv('~/Downloads/misclassified (3).csv')
    misclassified = misclassified.rename(columns={'Unnamed: 0': 'roi-id'})

    manifest = read_jsonlines(manifest_path)
    manifest = [x for x in manifest]
    slc = [x for x in manifest if int(x['roi-id']) < 3e6]
    roi_ids = [x['roi-id'] for x in slc]

    disagreement, labeler_stats_total = calc_disagreement(manifest=slc, roi_ids=roi_ids)
    print('TOTAL')
    print(f'disagreement: {disagreement}')
    print('\n')

    disagreement, labeler_stats_misclassified = calc_disagreement(manifest=slc, roi_ids=misclassified['roi-id'])
    print('MISCLASSIFIED')
    print(f'disagreement: {disagreement}')
    print('\n')

    df = combine_labeler_stats(stats_total=labeler_stats_total, stats_misclassified=labeler_stats_misclassified)
    print(df)


if __name__ == '__main__':
    main()

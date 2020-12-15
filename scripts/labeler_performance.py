import pandas as pd
import plotly.graph_objects as go
import numpy as np

from croissant.utils import read_jsonlines
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve

from util import get_experiment_genotype_map

project_name = 'ophys-experts-slc-oct-2020_ophys-experts-go-big-or-go-home'
manifest_path = 's3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020_behavior_3cre_1600roi_merged/output.manifest'

manifest = read_jsonlines(manifest_path)
manifest = [x for x in manifest]


def get_worker_ids(experiment_genotype_map, cre_line):
    worker_ids = set()
    for x in manifest:
        if not experiment_genotype_map[x['experiment-id']][:3] == cre_line:
            continue
        workers = x[project_name]['workerAnnotations']

        for worker in workers:
            worker_ids.add(worker['workerId'])
    return worker_ids


def get_labels(experiment_genotype_map, cre_line):
    worker_ids = get_worker_ids(experiment_genotype_map=experiment_genotype_map, cre_line=cre_line)

    res = {'label': []}
    for worker_id in worker_ids:
        res[worker_id] = []

    for x in manifest:
        if not experiment_genotype_map[x['experiment-id']][:3] == cre_line:
            continue

        workers = x[project_name]['workerAnnotations']
        label = x[project_name]['majorityLabel']
        res['label'].append(label)
        for id in worker_ids:
            annotation = [x for x in workers if x['workerId'] == id]
            if not annotation:
                res[id].append(np.nan)
            else:
                res[id].append(annotation[0]['roiLabel'])

    df = pd.DataFrame(res)
    return df


def calc_worker_precision(worker_id, df):
    TP = ((df['label'] == 'cell') & (df[worker_id] == 'cell')).sum()
    FP = ((df['label'] == 'not cell') & (df[worker_id] == 'cell')).sum()
    return TP / (TP + FP)


def calc_worker_recall(worker_id, df):
    TP = ((df['label'] == 'cell') & (df[worker_id] == 'cell')).sum()
    FN = ((df['label'] == 'cell') & (df[worker_id] == 'not cell')).sum()
    return TP / (TP + FN)


def plot_model_vs_labeler_performance_for_creline(fig, cre_line, row, col, showlegend=True):
    labeler = calc_labeler_perf_for_cre_line(cre_line=cre_line)
    model = pd.read_csv(f'~/Downloads/cnn_{cre_line.lower()}_pr_curve.csv')

    fig.add_trace(go.Scatter(x=labeler['recall'], y=labeler['precision'], showlegend=showlegend,
                             mode='markers',
                             name='Labeler',
                             marker=dict(
                                 color='blue'
                             )),
                  row=row,
                  col=col)
    fig.add_trace(go.Scatter(x=model['recall'], y=model['precision'], showlegend=showlegend,
                             mode='lines',
                             name='CNN',
                             marker=dict(
                                 color='green'
                             )),
                  row=row,
                  col=col)
    fig.update_xaxes(range=[0, 1.01], row=row, col=col)
    fig.update_yaxes(range=[0, 1.01], row=row, col=col)


def calc_labeler_perf_for_cre_line(cre_line):
    experiment_genotype_map = get_experiment_genotype_map()

    worker_ids = get_worker_ids(experiment_genotype_map=experiment_genotype_map, cre_line=cre_line)
    labels = get_labels(experiment_genotype_map=experiment_genotype_map, cre_line=cre_line)

    precisions = []
    recalls = []
    label_percents = []

    print('\n')
    print(f'{cre_line}')

    for worker_id in worker_ids:
        p = calc_worker_precision(worker_id=worker_id, df=labels)
        r = calc_worker_recall(worker_id=worker_id, df=labels)
        precisions.append(p)
        recalls.append(r)

        label_percent = labels[worker_id].notnull().mean() * 100
        label_percents.append(label_percent)

        print(f'{worker_id}\t precision: {p:.2f}\t recall: {r:.2f}\t labeled: {label_percent:.1f}%')

    return pd.DataFrame({'precision': precisions, 'recall': recalls, 'Percent labeled': label_percents})


def main():
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Slc", "Sst", "Vip"), x_title='Recall', y_title='Precision',
                        row_heights=[300], column_widths=[300, 300, 300])
    fig.update_layout(title_text='Human level performance comparison', width=1600)

    plot_model_vs_labeler_performance_for_creline(fig=fig, cre_line='Slc', row=1, col=1, showlegend=True)
    plot_model_vs_labeler_performance_for_creline(fig=fig, cre_line='Sst', row=1, col=2, showlegend=False)
    plot_model_vs_labeler_performance_for_creline(fig=fig, cre_line='Vip', row=1, col=3, showlegend=False)

    fig.write_image('labeler_performance.png')


if __name__ == '__main__':
    main()

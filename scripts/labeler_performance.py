import pandas as pd
import numpy as np

from croissant.utils import read_jsonlines

manifest = read_jsonlines(uri='/Users/adam.amster/Downloads/output.manifest')
manifest = [x for x in manifest]

def get_worker_ids():
    worker_ids = set()
    for x in manifest:
        workers = x['ophys-experts-slc-oct-2020']['workerAnnotations']

        for worker in workers:
            worker_ids.add(worker['workerId'])
    return worker_ids

def get_labels():
    worker_ids = get_worker_ids()
    res = {'label': []}
    for worker_id in worker_ids:
        res[worker_id] = []

    for x in manifest:
        workers = x['ophys-experts-slc-oct-2020']['workerAnnotations']
        label = x['ophys-experts-slc-oct-2020']['majorityLabel']
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

def main():
    worker_ids = get_worker_ids()
    labels = get_labels()

    for worker_id in worker_ids:
        p = calc_worker_precision(worker_id=worker_id, df=labels)
        r = calc_worker_recall(worker_id=worker_id, df=labels)
        label_frac = labels[worker_id].notnull().mean()
        print(f'{worker_id}\t precision: {p:.2f}\t recall: {r:.2f}\t labeled: {label_frac*100:.1f}%')

if __name__ == '__main__':
    main()



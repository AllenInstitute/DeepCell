import os

import matplotlib.pyplot as plt
import json
import skimage.color
import pandas as pd
import numpy as np
from pathlib import Path
import slapp.utils.query_utils as qu
import visual_behavior.data_access.loading as loading
from matplotlib.backends.backend_pdf import PdfPages

mapped_drive = '/Users/adam.amster/ibs-adama-ux3'

# NOTE requires inference-plots env (dependency conflicts)

def get_LIMS_data(dbconn, exp_id):
    """get the rois, image projection paths from LIMS
    """
    results = {}
    results['cell_rois'] = dbconn.query(
            f"SELECT * FROM cell_rois WHERE ophys_experiment_id={exp_id}")
    oeq = dbconn.query(
            f"SELECT * FROM ophys_experiments WHERE id={exp_id}")[0]
    sdir = Path(f"{mapped_drive}/{oeq['storage_directory']}")
    shape = (oeq['movie_width'], oeq['movie_height'])
    results['max_int_mask_image'] = str(next(sdir.rglob("maxInt_a13a.png")))
    results['ophys_average_intensity_projection_image'] = str(next(
            sdir.rglob("avgInt_a1X.png")))
    results['id'] = exp_id
    results['stimulus_name'] = ""
    return results, shape


def rois_on_im(rois, im):
    im = skimage.color.gray2rgb(im)
    for roi in rois:
        submask = np.array(roi['mask_matrix']).astype('uint8') * 255
        r0 = roi['y']
        r1 = roi['y'] + roi['height']
        c0 = roi['x']
        c1 = roi['x'] + roi['width']
        im[:, :, 0][r0:r1, c0:c1][submask != 0] = submask[submask != 0] * 0.5
    return im


def reshape_mask(roi, im):
    submask = np.array(roi['mask_matrix'])
    mask = np.zeros(im.shape)
    r0 = roi['y']
    r1 = roi['y'] + roi['height']
    c0 = roi['x']
    c1 = roi['x'] + roi['width']
    mask[r0:r1, c0:c1] = submask
    return mask




def get_experiment_metadata(experiment_id):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = Path(root) / 'ophys_metadata_lookup.txt'
    df = pd.read_csv(path)
    metadata = df[df['experiment_id'] == experiment_id].iloc[0]
    return metadata.to_dict()


def run_for_experiment(experiment_id):
    # get max projection image
    movie_path = loading.get_motion_corrected_movie_h5_location(experiment_id)
    max_proj_path = Path(movie_path).parent / f'{experiment_id}_maximum_projection.png'
    max_proj_path = f'{mapped_drive}/{max_proj_path}'
    max_proj = plt.imread(max_proj_path)

    # get LIMS results
    creds = qu.get_db_credentials(env_prefix="LIMS_", **qu.lims_defaults)
    conn = qu.DbConnection(**creds)
    lims_data, shape = get_LIMS_data(conn, experiment_id)
    lims_rois = lims_data['cell_rois']
    valid_prod_rois = [r for r in lims_rois if r['valid_roi']]
    invalid_prod_rois = [r for r in lims_rois if not r['valid_roi']]

    # get binarized rois and labels
    rois_path = f"{mapped_drive}/allen/aibs/informatics/danielk/dev_LIMS/new_labeling/{experiment_id}/binarize_output.json"
    with open(rois_path, "r") as f:
        cnn_rois = json.load(f)
    labels_path = "/Users/adam.amster/Downloads/inference.csv"
    labels = pd.read_csv(labels_path)
    labels = labels[labels['experiment_id'] == experiment_id]
    label_map = {i['roi-id']: i['y_score'] > .4 for _, i in labels.iterrows()}

    cnn_rois = [i for i in cnn_rois if i['id'] in label_map]
    valid_cnn_rois = [r for r in cnn_rois if label_map[r['id']]]
    invalid_cnn_rois = [r for r in cnn_rois if not label_map[r['id']]]

    im_prod_valid = rois_on_im(valid_prod_rois, max_proj)
    im_prod_invalid = rois_on_im(invalid_prod_rois, max_proj)
    im_cnn_valid = rois_on_im(valid_cnn_rois, max_proj)
    im_cnn_invalid = rois_on_im(invalid_cnn_rois, max_proj)

    f, a = plt.subplots(2, 2, clear=True, sharex=True, sharey=True, num=1)
    a[0, 0].imshow(im_prod_valid)
    a[0, 0].set_title(f"{len(valid_prod_rois)} Production Cells", fontsize=8)
    a[0, 1].imshow(im_prod_invalid)
    a[0, 1].set_title(f"{len(invalid_prod_rois)} Production NOT Cells", fontsize=8)
    a[1, 0].imshow(im_cnn_valid)
    a[1, 0].set_title(f"{len(valid_cnn_rois)} Suite2P + CNN Cells", fontsize=8)
    a[1, 1].imshow(im_cnn_invalid)
    a[1, 1].set_title(f"{len(invalid_cnn_rois)} Suite2P + CNN NOT Cells", fontsize=8)
    _ = [ia.grid(alpha=0.7) for ia in a.flat]

    # add metadata
    metadata = get_experiment_metadata(experiment_id=experiment_id)
    props = dict(boxstyle='round', facecolor='wheat', pad=1)
    textstr = f'''
    Experiment: {metadata["experiment_id"]}
    Depth: {metadata["depth"]}
    Rig: {metadata["rig"]}
    Cre Line: {metadata["genotype"][:3]}
    '''
    plt.subplots_adjust(right=.7)

    f.text(0.85, .75, textstr, bbox=props, fontsize=6, ha='center')
    return f


if __name__ == '__main__':
    inference_res = pd.read_csv('~/Downloads/inference.csv')
    experiments = inference_res['experiment_id'].unique()
    experiments = [904352688]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = Path(root) / 'inference-output' / 'inference_plots.pdf'
    with PdfPages(f'{path}') as pdf:
        for i, experiment_id in enumerate(experiments):
            print(f'experiment {i}')
            f = run_for_experiment(experiment_id=experiment_id)
            pdf.savefig(f)
            plt.close()
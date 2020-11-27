import matplotlib.pyplot as plt

from SlcDataset import SlcDataset
from util import get_random_roi


def display_roi(mask, max_, avg_, roi_id, columns, rows, label, y_pred=None):
    imgs = [mask, max_, avg_]
    type_ = ['mask', 'max', 'avg']
    fig = plt.figure(figsize=(20, 20))
    for i in range(1, columns * rows + 1):
        title = f'{label} {roi_id} {type_[i - 1]}'
        if y_pred is not None:
            title += f' {y_pred}'
        fig.add_subplot(rows, columns, i, title=title)
        plt.imshow(imgs[i - 1], cmap='gray', vmin=0, vmax=255)
    plt.show()


def display_differently_labeled_rois(data: SlcDataset):
    cell_mask, cell_max, cell_avg, cell_roi_id = get_random_roi(data=data, label=1)
    display_roi(mask=cell_mask, max_=cell_max, avg_=cell_avg, roi_id=cell_roi_id, rows=1, columns=3)

    not_cell_mask, not_cell_max, not_cell_avg, not_cell_roi_id = get_random_roi(data=data, label=0)
    display_roi(mask=not_cell_mask, max_=not_cell_max, avg_=not_cell_avg, roi_id=not_cell_roi_id, rows=1, columns=3)


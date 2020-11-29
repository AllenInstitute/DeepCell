import matplotlib.pyplot as plt

from SlcDataset import SlcDataset
from util import get_random_roi


def display_roi(data: SlcDataset, roi_id, columns, rows, label, y_pred=None, channels=None):
    if channels is None:
        data, _ = data[data.roi_ids.index(roi_id)]
        channels = data
    avg_, max_, mask = channels[:, :, 0], channels[:, :, 1], channels[:, :, 2]
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
    cell_roi_id = get_random_roi(data=data, label=1)
    display_roi(data=data, roi_id=cell_roi_id, rows=1, columns=3, label='Cell')

    not_cell_roi_id = get_random_roi(data=data, label=0)
    display_roi(data=data, roi_id=not_cell_roi_id, rows=1, columns=3, label='Not Cell')


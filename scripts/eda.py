import matplotlib.pyplot as plt

from deepcell.roi_dataset import RoiDataset


def display_roi(roi_id, columns, rows, label, y_pred=None, channels=None, data: RoiDataset = None):
    if channels is None and data is None:
        raise ValueError('Need to supply either dataset or numpy array')

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


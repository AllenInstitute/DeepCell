import tempfile

import numpy as np
import pytest
from PIL import Image

from deepcell.datasets.channel import Channel
from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.testing.util import get_test_data


class TestROIDataset:

    @pytest.mark.parametrize('channels', (
        [Channel.MASK],
        [Channel.MASK, Channel.MAX_PROJECTION],
        [Channel.MASK, Channel.MAX_PROJECTION, Channel.CORRELATION_PROJECTION]
    ))
    def test_construct_input(self, channels):
        """Tests that we can pass an arbitrary list of channels to construct,
        and it is constructed correctly"""
        n_rois = 5
        with tempfile.TemporaryDirectory() as tmpdir:
            get_test_data(
                write_dir=tmpdir,
                exp_id='0',
                n_rois=n_rois
            )
            model_inputs = [
                ModelInput.from_data_dir(
                    data_dir=tmpdir,
                    channels=channels,
                    experiment_id='0',
                    roi_id=str(roi_id)
                )
                for roi_id in range(n_rois)]
            dataset = RoiDataset(
                model_inputs=model_inputs
            )

            obs = model_inputs[0]

            imgs = {c: np.array(Image.open(p))
                    for c, p in obs.channel_path_map.items()}
            obs = dataset._construct_input(
                obs=obs
            )
            assert obs.shape[-1] == len(channels)
            for i, c in enumerate(channels):
                np.testing.assert_allclose(obs[:, :, i], imgs[c])

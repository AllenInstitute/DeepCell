import tempfile
from pathlib import Path

import pytest

from deepcell.datasets.channel import Channel, channel_filename_prefix_map
from deepcell.datasets.model_input import ModelInput
from deepcell.testing.util import get_test_data


class TestModelInput:
    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._exp_id = '0'

        get_test_data(
            write_dir=cls._tmpdir.name,
            exp_id=cls._exp_id
        )

    @classmethod
    def teardown_class(cls):
        cls._tmpdir.cleanup()

    @pytest.mark.parametrize('channels', (
            [Channel.MASK, Channel.MAX_PROJECTION, Channel.AVG_PROJECTION],
            [Channel.MASK, Channel.MAX_PROJECTION,
             Channel.CORRELATION_PROJECTION]
    ))
    @pytest.mark.parametrize('roi_id', ('0', '99'))
    def test_from_data_dir(self, channels, roi_id):
        expected_channel_path_map = {
            c: Path(self._tmpdir.name) /
            f'{channel_filename_prefix_map[c]}_'
            f'{self._exp_id}_{roi_id}.png'
            for c in channels
        }
        if roi_id == '99':
            # doesn't exist
            with pytest.raises(ValueError):
                ModelInput.from_data_dir(
                    data_dir=self._tmpdir.name,
                    experiment_id=self._exp_id,
                    roi_id=roi_id,
                    channels=channels
                )
        else:
            model_input = ModelInput.from_data_dir(
                data_dir=self._tmpdir.name,
                experiment_id=self._exp_id,
                roi_id=roi_id,
                channels=channels
            )
            assert model_input.channel_path_map == {
                channel: expected_channel_path_map[channel]
                for channel in model_input.channel_path_map}

    @pytest.mark.parametrize('channel_order', (
        [Channel.MASK, Channel.MAX_PROJECTION],
        [Channel.MASK, Channel.AVG_PROJECTION]
    ))
    def test_validate_channel_order(self, channel_order):
        channel_path_map = {
            Channel.MASK: Path('foo'),
            Channel.MAX_PROJECTION: Path('foo')
        }
        if channel_order == [Channel.MASK, Channel.MAX_PROJECTION]:
            ModelInput._validate_channel_order(
                channel_order=channel_order,
                channel_path_map=channel_path_map
            )
        else:
            # invalid
            with pytest.raises(ValueError):
                ModelInput._validate_channel_order(
                    channel_order=channel_order,
                    channel_path_map=channel_path_map
                )

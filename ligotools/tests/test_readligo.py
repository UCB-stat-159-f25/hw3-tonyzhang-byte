from pathlib import Path

import pytest

from ligotools import readligo as rl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
H1_FILE = DATA_DIR / "H-H1_LOSC_4_V2-1126259446-32.hdf5"


@pytest.mark.filterwarnings("ignore:.*")
def test_loaddata_h1_returns_expected_lengths_and_dt():
    strain, time, chan_dict = rl.loaddata(str(H1_FILE), "H1")

    assert len(strain) == len(time) == 131072
    assert time[1] - time[0] == pytest.approx(1 / 4096)
    assert chan_dict["DATA"].sum() == 32


@pytest.mark.filterwarnings("ignore:.*")
def test_dq2segs_identifies_expected_segment():
    _, time, chan_dict = rl.loaddata(str(H1_FILE), "H1")

    segments = rl.dq2segs(chan_dict, int(time[0]))

    assert len(segments.seglist) == 1
    segment_start, segment_stop = segments.seglist[0]
    assert segment_start == 1126259446
    assert segment_stop == 1126259478

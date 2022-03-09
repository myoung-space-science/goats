import pathlib

from goats.core import datasets


def test_basic(rootpath: pathlib.Path):
    """Test the basic dataset interface."""
    datapath = rootpath / 'basic_dataset.nc'
    dataset = datasets.DatasetView(datapath)
    assert isinstance(dataset.variables, datasets.DataViewer)
    assert isinstance(dataset.sizes, datasets.DataViewer)
    assert isinstance(dataset.axes, tuple)


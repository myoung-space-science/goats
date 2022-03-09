import collections.abc
import configparser
import json
import pathlib

from goats.core import iotools


# read version from installed package
from importlib.metadata import version
__version__ = version("goats")


class Environment(collections.abc.Mapping):
    """A collection of environmental settings."""

    _default = {
        'path': pathlib.Path(__file__).parent / 'goats.ini',
    }
    """Internal class-wide default attribute values."""

    def __init__(self, name: str) -> None:
        self.name = name
        """The name of the observing package to select."""
        self._package = f"{__package__}.{self.name}"
        path = self._default['path']
        self._config = self._get_package_config(path)

    def __len__(self) -> int:
        """The number of available parameter values."""
        return len(self._config)

    def __iter__(self):
        """Iterate over available parameter values."""
        yield from self._config

    def __contains__(self, key):
        """True if `key` names an available parameter."""
        return key in self._config

    def __getitem__(self, key: str):
        """Access parameter values by mapping key."""
        if key in self._config:
            return self._config[key]
        raise KeyError(
            f"{self._package} has no value for {key!r}"
        ) from None

    def _get_package_config(self, path: iotools.PathLike):
        """Get the package-specific configuration parameters."""
        config = configparser.ConfigParser()
        config.read(iotools.ReadOnlyPath(path))
        return config[self.name]

    def __str__(self) -> str:
        return json.dumps(
            dict(self._config),
            indent=4,
            sort_keys=True,
        )

    def __repr__(self) -> str:
        return f"{self._package}:\n{self}"

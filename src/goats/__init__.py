import collections.abc
import configparser
import json
import os
import pathlib

from goats.core import iotools


# read version from installed package
from importlib.metadata import version
__version__ = version("goats")


class Environment(collections.abc.Mapping):
    """A collection of environmental settings."""

    def __init__(self, name: str) -> None:
        self.name = name
        """The name of the observing package to select."""
        self._package = f"{__package__}.{self.name}"
        home = pathlib.Path('~').expanduser()
        paths = [
            pathlib.Path.cwd(), # The current working directory
            home, # The user's home directory
            home / '.config', # Linux standard (local)
            '/etc/goats', # Linux standard (global)
            os.environ.get('GOATS_INI'), # A known environment variable
            pathlib.Path(__file__).parent, # The package top
        ]
        config = configparser.ConfigParser()
        path = iotools.search(paths, 'goats.ini')
        config.read(iotools.ReadOnlyPath(path))
        self._config = config[self.name]
        self.path = path

    def __len__(self) -> int:
        """The number of available parameter values."""
        return len(self._config)

    def __iter__(self):
        """Iterate over available parameter values."""
        yield from self._config

    def __getitem__(self, key: str):
        """Access parameter values by mapping key."""
        if key in self._config:
            return self._config[key]
        raise KeyError(
            f"{self._package} has no value for {key!r}"
        ) from None

    def __str__(self) -> str:
        return json.dumps(
            dict(self._config),
            indent=4,
            sort_keys=True,
        )

    def __repr__(self) -> str:
        return f"{self._package}({self.path}):\n{self}"

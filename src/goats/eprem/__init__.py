from .. import iotools
from ._observing import Observer, _pkg
from .parameters import BaseTypesH


__all__ = [
    'basetypes',
    'Stream',
    'Point',
]


basetypes = BaseTypesH(source=_pkg['src'])


class Stream(Observer):
    """An EPREM stream observer."""

    def __init__(
        self,
        name: int=None,
        path: iotools.PathLike=None,
        config: iotools.PathLike=None,
        system: str='mks'
    ) -> None:
        templates = [
            lambda n: f'obs{n:06}.nc',
            lambda n: f'flux{n:06}.nc',
        ]
        super().__init__(
            templates,
            name=name,
            path=path,
            config=config,
            system=system
        )


class Point(Observer):
    """An EPREM point observer."""

    def __init__(
        self,
        name: int=None,
        path: iotools.PathLike=None,
        config: iotools.PathLike=None,
        system: str='mks'
    ) -> None:
        templates = [
            lambda n: f'p_obs{n:06}.nc',
        ]
        super().__init__(
            templates,
            name=name,
            path=path,
            config=config,
            system=system
        )


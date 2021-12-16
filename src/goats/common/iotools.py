import os
import pathlib
from typing import *


class SelectiveSingleton:
    """A base object for creating classes with special singleton instances.

    Subclasses of this class must provide an iterable of cases to treat as
    singletons. The given iterable may be empty. At instance creation, this
    class will check for an existing instance with the given key. If it finds an
    existing instance it will return that instance; otherwise, it will create a
    new instance, store the instance if the key corresponds to a singleton case,
    and return the instance. Note that the key must be hashable.
    """
    _instances = {}
    _singletons = None
    def __new__(cls, key: Hashable, *args, **kwargs) -> Any:
        """Create a new instance or return an existing instance."""
        if cls._singletons is None:
            message = f"Singleton cases must be iterable, not {cls._singletons}"
            raise NotImplementedError(message) from None
        if key in cls._instances:
            return cls._instances[key]
        new = super(SelectiveSingleton, cls).__new__(cls, *args, **kwargs)
        if key in cls._singletons:
            cls._instances[key] = new
        return new


class SingleInstance(type):
    """A metaclass for defining singleton-like objects.

    Using this class as the metaclass for a class that manages path-based
    objects (e.g., datasets) will ensure that only one instance of the object
    exists. This may be useful when multiple other objects need to access an
    instance of the class given a common path. This metaclass will simply return
    the existing instance instead of creating a new one.

    See https://refactoring.guru/design-patterns/singleton/python/example
    """
    _instances = {}
    def __call__(cls, path: Union[str, pathlib.Path], *args, **kwargs):
        """Ensure that only one instance of the given object exists."""
        if path not in cls._instances:
            cls._instances[path] = super(
                SingleInstance, cls
            ).__call__(path, *args, **kwargs)
        return cls._instances[path]


class NonExistentPathError(Exception):

    def __init__(self, path: str=None):
        self._path = path

    @property
    def path(self) -> str:
        if self._path is None:
            self._path = "The requested path"
        return self._path

    def __str__(self):
        return f"{self.path} does not exist."


class ReadOnlyPathError(Exception):

    def __init__(self, obj: object):
        self._obj = obj

    def __str__(self):
        return f"Objects of type {self._obj} are read-only."


class ReadOnlyPath(pathlib.Path):
    """A wrapper for read-oriented paths.
    
    This class creates ``pathlib.Path`` objects intended for reading. The instance path is fully resolved with the user wildcard expanded, and raises an exception if the requested path does not exist. 
    
    This class overloads the following methods to prevent the user from writing to path:
    ```
        pathlib.Path().write_bytes()
        pathlib.Path().write_text()
        pathlib.Path().open(mode='w')
    ```
    It does this in lieu of changing path permissions, which may be undesirable.
    """

    def __new__(cls, *args, **kwargs):
        """Create a new path object of the appropriate type."""
        _type = (
            pathlib.WindowsPath if os.name == 'nt'
            else pathlib.PosixPath
        )
        path = super().__new__(_type, *args, **kwargs)
        inst = path.expanduser().resolve()
        if not inst.exists():
            raise NonExistentPathError(inst)
        return inst

    def open(self, *args, **kwargs):
        if 'w' in args:
            raise ReadOnlyPathError(self.__class__)
        return super().open(*args, **kwargs)

    def write_bytes(self, *args, **kwargs):
        raise ReadOnlyPathError(self.__class__)

    def write_text(self, *args, **kwargs):
        raise ReadOnlyPathError(self.__class__)


def file_lines(file: Union[str, pathlib.Path]) -> int:
    """Count the number of lines in a file."""
    with pathlib.Path(file).open('r') as fp:
        nlines = len(fp.readlines())
    return nlines


def strip_inline_comments(string: str, comments: List[str]) -> str:
    """Remove inline comments from a string."""
    for c in comments:
        parts = string.split(c)
        string = parts[0]
    return string.strip()


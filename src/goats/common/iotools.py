import os
import pathlib
import typing

from goats.common import iterables


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
    def __new__(cls, key: typing.Hashable, *args, **kwargs) -> typing.Any:
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
    def __call__(cls, path: typing.Union[str, pathlib.Path], *args, **kwargs):
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


Parsable = typing.TypeVar('Parsable')
Parsable = typing.Dict[str, typing.Any]


class TextFile(iterables.ReprStrMixin):
    """A representation of a text file with pattern-based search."""

    def __init__(self, path: typing.Union[str, pathlib.Path]) -> None:
        """
        Parameters
        ----------
        path : string or `pathlib.Path`
            The path to the target text file. May be relative and contain
            wildcards.
        """
        self._path = path
        self._lines = None

    @property
    def lines(self):
        """The lines of text in this file."""
        if self._lines is None:
            with self.path.open('r') as fp:
                self._lines = fp.readlines()
        return self._lines

    @property
    def path(self):
        """The fully resolved, read-only path."""
        return ReadOnlyPath(self._path)

    KT = typing.TypeVar('KT', bound=typing.Hashable)
    VT = typing.TypeVar('VT')
    Matched = typing.TypeVar('Matched')
    Parsed = typing.TypeVar('Parsed', bound=tuple)
    Parsed = typing.Tuple[KT, VT]
    def extract(
        self,
        match: typing.Callable[[str], Matched],
        parse: typing.Callable[[Matched], Parsed],
    ) -> typing.Dict[KT, VT]:
        """Search each line and parse those that meet given criteria.
        
        Parameters
        ----------
        match : callable
            A callable object that takes a string and returns matches to a
            pattern.
        parse : callable
            A callable object that takes the output of `match`, and returns a
            tuple containing a valid mapping key and corresponding value. It may
            assume that the input is not empty.

        Returns
        -------
        dict
            A dictionary constructed from the tuples output by `parse`.
        """
        matches = [match(line) for line in self.lines]
        parsed = [parse(match) for match in matches if match]
        return dict(parsed)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.path)


def file_lines(file: typing.Union[str, pathlib.Path]) -> int:
    """Count the number of lines in a file."""
    with pathlib.Path(file).open('r') as fp:
        nlines = len(fp.readlines())
    return nlines


def strip_inline_comments(string: str, comments: typing.List[str]) -> str:
    """Remove inline comments from a string."""
    for c in comments:
        parts = string.split(c)
        string = parts[0]
    return string.strip()


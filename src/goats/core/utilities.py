import typing

T = typing.TypeVar('T')


def getattrval(
    __object: T,
    __name: str,
    *args,
    **kwargs
) -> typing.Union[typing.Any, T]:
    """Get an appropriate value based on the given object type.
    
    Parameters
    ----------
    __object : Any
        The object from which to retrieve the target attribute, if available.

    __name : string
        The name of the target attribute.

    *args
        Optional positional arguments to pass to the target attribute, if it is
        callable.

    **kwargs
        Optional keyword arguments to pass to the target attribute, if it is
        callable.

    Returns
    -------
    Any
        The value of the attribute on the given object, or the object itself.
        See Notes for further explanation.

    Notes
    -----
    This function will attempt to retrieve the named attribute from the given
    object. If the attribute exists and is callable (e.g., a class method), this
    function will call the attribute with `*args` and `**kwargs`, and return the
    result. If the attribute exists and is not callable, this function will
    return it as-is. If the attribute does not exist, this function will return
    the given object. This case supports programmatic use when the calling code
    does not know the type of object until runtime.

    Examples
    --------
    TODO
    """
    attr = getattr(__object, __name, __object)
    return attr(*args, **kwargs) if callable(attr) else attr


@typing.overload
def setattrval(__object: T, __name: str, __value) -> None: ...


@typing.overload
def setattrval(__object: T, __name: str, __value, *args, **kwargs) -> None: ...


@typing.overload
def setattrval(__object: T, __name: str, *args, **kwargs) -> None: ...


def setattrval(*args, **kwargs):
    """Set an appropriate value based on the given object type.
    
    Parameters
    ----------
    __object : Any
        The object on which to set the target attribute.

    __name : string
        The name of the target attribute.

    __value : Any
        The new value of the target attribute.

    *args
        Positional arguments to pass the target attribute, if it is callable.
        See Notes for further explanation.

    **kwargs
        Keyword arguments to pass to the target attribute, if it is callable.
        See Notes for further explanation.

    Returns
    -------
    None

    Notes
    -----
    This function will attempt to set the named attribute on the given object.
    If the attribute exists and is callable (e.g., a class method), this
    function will call the attribute with all positional arguments after
    `__object` and `__name`, as well as any given keyword arguments. The user
    may pass the new value as the first positional argument or as a keyword
    argument, in order to support as many forms of callable attributes as
    possible. If the attribute exists and is not callable, this function will
    set the new value from the first positional argument after `__object` and
    `__name`. If the attribute does not exist, this function will raise an
    ``AttributeError``.

    Examples
    --------
    TODO
    """
    obj, name, *args = args
    attr = getattr(obj, name)
    if callable(attr):
        attr(*args, **kwargs)
    else:
        setattr(obj, name, args[0])


def equal_attrs(name: str, *objects):
    """True if all objects have the named attribute with equal values."""
    if len(objects) == 0:
        raise ValueError("No objects to compare")
    try:
        v = getattr(objects[0], name)
    except AttributeError:
        return False
    if len(objects) == 1:
        return True
    try:
        truth = all(getattr(i, name) == v for i in objects[1:])
    except AttributeError:
        return False
    return truth



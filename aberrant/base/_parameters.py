"""Constructor parameters shared by models, transformers and forecasters."""

import copy
import inspect
from typing import Any, ClassVar, Self

from aberrant import __version__
from aberrant.exceptions import IncompatibleStateError

VERSION_KEY = "__aberrant_version__"


def versioned_state(state: dict[str, Any]) -> dict[str, Any]:
    """``state`` stamped with the running aberrant version."""
    return {**state, VERSION_KEY: __version__}


def checked_state(component: object, state: dict[str, Any]) -> dict[str, Any]:
    """``state`` without its stamp, if this aberrant version saved it.

    Raises:
        IncompatibleStateError: Another version, or a release before 2.0,
            saved the state.
    """
    state = dict(state)
    saved = state.pop(VERSION_KEY, None)
    if saved != __version__:
        raise IncompatibleStateError(type(component).__name__, saved, __version__)
    return state


def _constructor_parameters(cls: type) -> list[inspect.Parameter]:
    return [
        parameter
        for parameter in inspect.signature(cls).parameters.values()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]


def _fresh(value: object, memo: dict[int, Any]) -> object:
    clone = getattr(value, "clone", None)
    if isinstance(value, Parameterized) and callable(clone):
        return clone()
    if isinstance(value, list):
        return [_fresh(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_fresh(item, memo) for item in value)
    if isinstance(value, dict):
        return {key: _fresh(item, memo) for key, item in value.items()}
    return copy.deepcopy(value, memo)


class Parameterized:
    """
    Parameters, cloning, reset and ``repr`` derived from ``__init__``.

    Every constructor parameter is stored on the instance under its own name,
    so ``get_params()`` returns exactly what rebuilds an equivalent, untrained
    component. ``clone()`` builds that component, ``reset()`` turns this
    instance into it, and ``repr`` shows the parameters that differ from their
    defaults.

    Pickles carry the aberrant version that wrote them, and only that version
    restores them. Attributes named in ``_transient`` are caches: pickles store
    ``None`` in their place.
    """

    _transient: ClassVar[tuple[str, ...]] = ()

    def __getstate__(self) -> dict[str, Any]:
        """The instance state, without caches, stamped with the version."""
        state = self.__dict__.copy()
        for name in self._transient:
            state[name] = None
        return versioned_state(state)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state saved by this aberrant version."""
        self.__dict__.update(checked_state(self, state))

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters of this instance by name."""
        return {
            parameter.name: getattr(self, parameter.name)
            for parameter in _constructor_parameters(type(self))
        }

    def clone(self) -> Self:
        """
        Return a new, untrained instance with the same parameters.

        Parameters that are components themselves are cloned too; other values
        are deep-copied, so the clone shares no state with this instance.
        """
        memo: dict[int, Any] = {}
        params = {
            name: _fresh(value, memo) for name, value in self.get_params().items()
        }
        return type(self)(**params)

    def reset(self) -> None:
        """Forget everything learned, as if this instance were just built."""
        fresh = self.clone()
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)

    def __repr__(self) -> str:
        """Show the class and every parameter that differs from its default."""
        shown = []
        for parameter in _constructor_parameters(type(self)):
            value = getattr(self, parameter.name)
            default = parameter.default
            if default is not parameter.empty and _same(value, default):
                continue
            shown.append(f"{parameter.name}={value!r}")
        return f"{type(self).__name__}({', '.join(shown)})"

    def __str__(self) -> str:
        """Return the same text as ``repr``."""
        return repr(self)


def _same(value: object, default: object) -> bool:
    if value is default:
        return True
    try:
        return bool(type(value) is type(default) and value == default)
    except (TypeError, ValueError):
        return False

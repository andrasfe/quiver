"""Quiver backends."""

from quivercirc.backends.base import Backend
from quivercirc.backends.numpy_backend import NumpyBackend

__all__ = ["Backend", "NumpyBackend"]


def pennylane_backend(*args, **kwargs):
    from quivercirc.backends.pennylane_backend import PennyLaneBackend

    return PennyLaneBackend(*args, **kwargs)


def qiskit_backend(*args, **kwargs):
    from quivercirc.backends.qiskit_backend import QiskitBackend

    return QiskitBackend(*args, **kwargs)

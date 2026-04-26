"""Quiver backends."""

from quiver.backends.base import Backend
from quiver.backends.numpy_backend import NumpyBackend

__all__ = ["Backend", "NumpyBackend"]


def pennylane_backend(*args, **kwargs):
    from quiver.backends.pennylane_backend import PennyLaneBackend

    return PennyLaneBackend(*args, **kwargs)


def qiskit_backend(*args, **kwargs):
    from quiver.backends.qiskit_backend import QiskitBackend

    return QiskitBackend(*args, **kwargs)

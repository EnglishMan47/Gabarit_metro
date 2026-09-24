"""Gabarit: obstacle detection in the train clearance envelope from 3D lidar.

The core package depends only on NumPy. ROS 2 and the web interface are thin
adapters around :class:`gabarit.pipeline.Pipeline`.
"""

import os as _os

# The algorithm works with many small matrices; multi-threaded BLAS makes them
# up to 100x slower because of thread start-up costs. Keep BLAS single-threaded
# (must happen before NumPy is imported; see also limit_blas_threads()).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "2.0.0"


def limit_blas_threads() -> None:
    """Best effort for the case NumPy was imported before this package."""
    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        threadpool_limits(1)
    except Exception:  # pragma: no cover - optional dependency
        pass

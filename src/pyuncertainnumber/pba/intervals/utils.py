import numpy as np


def safe_asarray(x):
    return (
        x
        if isinstance(x, np.ndarray) and x.dtype == float
        else np.asarray(x, dtype=float)
    )

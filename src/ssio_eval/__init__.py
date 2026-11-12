"""ssio_eval -- a small, explicit odometry evaluator.

Public surface:

    from ssio_eval import load_trajectory, evaluate

    est = load_trajectory("runs/est.csv")
    gt = load_trajectory("data/euroc/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv")
    res = evaluate(est, gt, align="se3")

Everything else (frames, traj, metrics) is importable but not required.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .io.datasets import load_trajectory  # noqa: E402
from .metrics import ate, ave, evaluate, rpe  # noqa: E402
from .traj import Trajectory  # noqa: E402

__all__ = ["Trajectory", "load_trajectory", "ate", "ave", "rpe", "evaluate", "__version__"]
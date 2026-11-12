"""Dump an SSIO run's predicted and ground-truth trajectories to CSV.

Why this exists: ``SSIO/evaluate.py`` prints metrics and plots curves but never
writes the trajectories, so there is no way to re-score a run with an
independent evaluator.  This script reproduces the evaluator's inference path
and writes two EuRoC-style CSVs (ground truth + prediction) that
``ssio_eval`` can read.

Cross-checking the two evaluators on the same numbers is the only honest way to
know whether ``docs/METRICS.md`` and the paper use the same convention.

Usage
-----
    python tools/dump_ssio_traj.py D:/MIU/SSIO/experiments/quickstart/pgd \
        --ssio-root D:/MIU/SSIO --out runs/quickstart

Outputs ``<out>/gt_<seq>.csv`` and ``<out>/pred_<seq>.csv`` with columns
``timestamp, p_x, p_y, p_z, v_x, v_y, v_z`` (metres, m/s, seconds).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hydra_dir", help="SSIO run directory (contains .hydra/config.yaml)")
    p.add_argument("--ssio-root", default=os.environ.get("SSIO_ROOT", "D:/MIU/SSIO"))
    p.add_argument("--out", default="runs/ssio_export")
    p.add_argument("--checkpoint", default=None, help="default: highest-step checkpoint")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ssio_root = Path(args.ssio_root).resolve()
    if not (ssio_root / "evaluate.py").is_file():
        raise SystemExit(f"not an SSIO checkout: {ssio_root}")

    hydra_dir = Path(args.hydra_dir).resolve()
    cfg_path = hydra_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"missing {cfg_path}")

    # The config uses custom OmegaConf resolvers (host_env / as_device / as_dtype)
    # that are registered as a side effect of importing Interface, so import
    # evaluate (which does that) before touching the config.
    # Resolve the output dir first: the chdir below would otherwise capture it.
    out_dir = Path(args.out).resolve()
    sys.path.insert(0, str(ssio_root))
    os.chdir(ssio_root)  # dataset paths are resolved relative to the repo root

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    import evaluate as ev  # noqa: PLC0415
    from Interface import instantiate  # noqa: PLC0415
    from Interface.data import IMUData_w_GT  # noqa: PLC0415
    from Interface.model import INetwork  # noqa: PLC0415
    from Interface.sequence import ISequence  # noqa: PLC0415

    cfg = OmegaConf.load(cfg_path)

    ckpt = args.checkpoint or ev.pick_checkpoint(str(hydra_dir), None)
    print(f"[dump] config    : {cfg_path}")
    print(f"[dump] checkpoint: {os.path.basename(ckpt)}")

    network = instantiate(INetwork, cfg.network).to(cfg.meta.device, cfg.meta.dtype)
    network.load_state_dict(torch.load(ckpt, map_location=cfg.meta.device))
    network.eval()

    use_aux_imu = bool(OmegaConf.select(cfg, "modality.use_aux_imu", default=True))
    use_joints = bool(OmegaConf.select(cfg, "modality.use_joints", default=True))
    gravity = float(OmegaConf.select(cfg, "train.gravity", default=9.81007))
    imu_input = str(OmegaConf.select(cfg, "train.dv_imu_input", default="corrected"))

    sequences = [instantiate(ISequence[IMUData_w_GT], sc) for sc in cfg.eval_dataset.sequences]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        name = seq.seq_name
        print(f"[dump] inferring {name} …", flush=True)
        pred_w, gt_w, _gt_rot, t_ns_w, *_ = ev.run_sequence(
            network, seq, use_aux_imu=use_aux_imu, use_joints=use_joints,
            gravity=gravity, imu_input=imu_input,
        )

        pred = pred_w.reshape(-1, 3)
        gt = gt_w.reshape(-1, 3)
        t_ns = t_ns_w.reshape(-1)
        t_s = (t_ns - t_ns[0]).to(torch.float64).numpy() * 1e-9

        # Origin-anchored trapezoidal integration — exactly what evaluate.py scores.
        pred_pos = ev.integrate_trapezoid(pred, t_ns).numpy()
        gt_pos = ev.integrate_trapezoid(gt, t_ns).numpy()

        header = "timestamp, p_x, p_y, p_z, v_x, v_y, v_z"
        safe = name.replace("/", "_")
        for tag, pos, vel in (("gt", gt_pos, gt.numpy()), ("pred", pred_pos, pred.numpy())):
            data = np.column_stack([t_s, pos, vel])
            path = out_dir / f"{tag}_{safe}.csv"
            np.savetxt(path, data, delimiter=", ", fmt="%.9f", header=header, comments="# ")
            print(f"[dump] wrote {path}  ({data.shape[0]} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Fit a local pixel-to-joint residual from observed paired displacements.

Input JSON: image_key, image_wh, target, arm, samples containing delta_uv (2)
and delta_q (7). delta_q is radians. This tool never moves the robot.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def fit(data):
    if data["arm"] not in ("left", "right"):
        raise ValueError("arm must be left or right")
    wh = np.asarray(data["image_wh"])
    if wh.shape != (2,) or np.any(wh <= 0):
        raise ValueError("image_wh must be positive width,height")
    q = np.asarray([sample["delta_q"] for sample in data["samples"]], dtype=float)
    uv = np.asarray([sample["delta_uv"] for sample in data["samples"]], dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or uv.shape != (len(q), 2) or len(q) < 10:
        raise ValueError("Use >=10 measured pairs: 7 joint displacements and 2 pixel displacements")
    if not np.isfinite(q).all() or not np.isfinite(uv).all():
        raise ValueError("Calibration samples must be finite")
    if np.linalg.matrix_rank(q) < 7:
        raise ValueError("Joint samples do not span 7 joints; a full local Jacobian cannot be identified")
    jacobian = np.linalg.lstsq(q, uv, rcond=None)[0].T  # 2x7, pixels/rad
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[-1] < 1e-6 or singular[0] / singular[-1] > 100:
        raise ValueError("Image Jacobian is degenerate or poorly conditioned")
    prediction = q @ jacobian.T
    rms = float(np.sqrt(np.mean((prediction - uv)**2)))
    signal_rms = float(np.sqrt(np.mean(uv**2)))
    if signal_rms < 1.0 or rms / signal_rms > 0.3:
        raise ValueError("Calibration displacement too small or fit error exceeds 30%; reacquire locally")
    return {
        "image_key": data["image_key"], "image_wh": data["image_wh"],
        "arm": data["arm"], "target": data["target"],
        "pixel_to_joint": np.linalg.pinv(jacobian).tolist(),
        "image_jacobian": jacobian.tolist(), "fit_rms_px": rms,
        "sample_count": len(q), "condition_number": float(singular[0] / singular[-1]),
        "note": "Local calibration only; validate on held-out measured displacements before enabling.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = fit(json.loads(args.samples.read_text()))
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output}; fit RMS {result['fit_rms_px']:.3f} px. Servo remains disabled.")

#!/usr/bin/env python3
"""
export_scaler.py
────────────────
Phase 4 → Phase 5 handoff utility.

Converts the sklearn scaler (scaler.pkl) produced in Phase 4 into a
scaler.json that the Rust inference engine reads at startup.

Usage:
    python export_scaler.py \
        --input  ../phase4/ml/models/scaler.pkl \
        --output ./models/scaler.json

Also demonstrates how to export the trained model to ONNX.
"""

import argparse
import json
import pickle
from pathlib import Path


def export_scaler(pkl_path: str, json_path: str) -> None:
    """Load a sklearn StandardScaler or MinMaxScaler and dump as JSON."""
    with open(pkl_path, "rb") as f:
        scaler = pickle.load(f)

    scaler_type = type(scaler).__name__.lower()
    # Normalise name to "standard" or "minmax"
    if "standard" in scaler_type:
        scaler_type = "standard"
    elif "minmax" in scaler_type:
        scaler_type = "minmax"
    else:
        scaler_type = "standard"

    payload = {
        "scaler_type": scaler_type,
        "mean_":  scaler.mean_.tolist()  if hasattr(scaler, "mean_")  else scaler.data_min_.tolist(),
        "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else scaler.scale_.tolist(),
    }

    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"✓  Scaler exported → {json_path}")
    print(f"   type   : {scaler_type}")
    print(f"   features: {len(payload['mean_'])}")


def export_model_to_onnx(
    model_pkl: str,
    onnx_path: str,
    n_features: int = 12,
) -> None:
    """Export a fitted sklearn Pipeline/model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("skl2onnx not installed; skipping ONNX export.")
        print("Install with: pip install skl2onnx")
        return

    with open(model_pkl, "rb") as f:
        model = pickle.load(f)

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    options = {type(model): {"zipmap": False}}  # Return arrays, not dicts

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options=options,
        target_opset=17,
    )

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✓  ONNX model exported → {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 → 5 scaler/model export")
    parser.add_argument("--scaler-pkl",  default="../phase4/ml/models/scaler.pkl")
    parser.add_argument("--scaler-json", default="./models/scaler.json")
    parser.add_argument("--model-pkl",   default="../phase4/ml/models/best_model.pkl")
    parser.add_argument("--onnx-path",   default="./models/best_model.onnx")
    parser.add_argument("--n-features",  type=int, default=12)
    args = parser.parse_args()

    export_scaler(args.scaler_pkl, args.scaler_json)
    export_model_to_onnx(args.model_pkl, args.onnx_path, args.n_features)
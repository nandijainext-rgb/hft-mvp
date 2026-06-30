#!/usr/bin/env python3
"""
Create a minimal ONNX model and scaler.json for testing the Rust signal engine.

The model accepts the same input shape the Rust code sends:
  input:  float_input   [batch, 12]

It returns deterministic sklearn-style classifier outputs:
  output: label         [1]       int64
          probabilities [1, 3]    float32
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

N_FEATURES = 12


def create_model() -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install onnx")
        write_scaler("identity")
        print("NOTE: You still need best_model.onnx.")
        return

    input_info = helper.make_tensor_value_info(
        "float_input",
        TensorProto.FLOAT,
        [None, N_FEATURES],
    )
    label_info = helper.make_tensor_value_info("label", TensorProto.INT64, [1])
    probabilities_info = helper.make_tensor_value_info(
        "probabilities",
        TensorProto.FLOAT,
        [1, 3],
    )

    label_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["label"],
        value=helper.make_tensor("label_value", TensorProto.INT64, [1], [1]),
    )
    probabilities_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["probabilities"],
        value=helper.make_tensor(
            "probabilities_value",
            TensorProto.FLOAT,
            [1, 3],
            [0.20, 0.60, 0.20],
        ),
    )

    graph = helper.make_graph(
        [label_node, probabilities_node],
        "hft_test_classifier",
        [input_info],
        [label_info, probabilities_info],
    )
    model = helper.make_model(
        graph,
        producer_name="hft_mvp_updated",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    onnx_path = MODELS_DIR / "best_model.onnx"
    onnx.save(model, onnx_path)
    print(f"OK ONNX model -> {onnx_path}")

    write_scaler("identity")

    print("\nAll done! Run the signal engine with:")
    print("  ONNX_MODEL_PATH=models/best_model.onnx cargo run --release")


def write_scaler(scaler_type: str) -> None:
    scaler_dict = {
        "scaler_type": scaler_type,
        "mean_": [0.0] * N_FEATURES,
        "scale_": [1.0] * N_FEATURES,
    }
    scaler_path = MODELS_DIR / "scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"OK Scaler JSON -> {scaler_path}")


if __name__ == "__main__":
    create_model()

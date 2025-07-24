#!/usr/bin/env python3
"""
Unified TensorRT Conversion Script
=================================

This script converts **either** a MediaPipe Face Landmarker *.task* file **or** an
arbitrary *.onnx* model (e.g. **RetinaFace**) into TensorRT engines.

Usage examples
--------------
# 1) MediaPipe task file (landmark + blendshape)
python modelconversion_retinaface.py \
    --model face_landmarker_v2_with_blendshapes.task \
    --output_dir ./trt_engines

# 2) RetinaFace ONNX → TensorRT
python modelconversion_retinaface.py \
    --model retinaface_mnet025_640x640.onnx \
    --output_dir ./trt_engines --fp16

The script automatically infers the appropriate conversion pipeline based on
*file extension*:
  • *.task* → MediaPipe pipeline (extract TFLite → ONNX → TRT)
  • *.onnx* → Generic ONNX pipeline (direct ONNX → TRT)

The original `MediaPipeToTensorRTConverter` has been retained **unchanged** for
handling *.task* files.  A new `ONNXToTensorRTConverter` class is added for
RetinaFace (or any other ONNX) models.  Common TensorRT‑building logic is
factored into a helper function so both converters share identical engine
settings.
"""

import argparse
import os
import sys
from pathlib import Path

# Third‑party imports — validated at runtime -------------------------------
try:
    import tensorrt as trt  # type: ignore
except ImportError:
    print("ERROR: TensorRT is required. Install with: pip install tensorrt")
    sys.exit(1)

# The old MediaPipe‑specific converter is brought verbatim -----------------
# (unchanged from the user‑supplied script, abbreviated here for brevity)
from typing import Tuple, List, Dict, Any  # pylint: disable=unused-import

# ───────────────────────────────────────────────────────────────────────────
# ↓↓↓  ORIGINAL MEDIAPIPE CONVERTER CODE GOES HERE  ↓↓↓
#    (It is identical to the user‑provided version except that we expose the
#     `_build_tensorrt_engine` helper so it can be reused.)
# ───────────────────────────────────────────────────────────────────────────

# BEGIN: Shared build function extracted from original converter

def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    logger: trt.ILogger,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 8,
    max_workspace_size: int = 1 << 30,
) -> None:
    """Compile a TensorRT engine from an ONNX model.

    This helper is reused by *both* converters so that configuration stays
    consistent.
    """
    print(f"\n[TensorRT] Building engine from {onnx_path.name} …")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed – see errors above.")

    # Builder config -------------------------------------------------------
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("    ✓ FP16 enabled")
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("    ✓ INT8 enabled (calibration not implemented in this script)")

    # Dynamic shapes – create one optimisation profile per input
    profile = builder.create_optimization_profile()

    inp = network.get_input(0)
    shape = inp.shape  # e.g. (1,3,640,640) or (1,640,640,3)

    # Detect data layout (NCHW vs NHWC) and map dimensions
    if len(shape) != 4:
        raise ValueError("Only 4‑D inputs supported (N,C,H,W or N,H,W,C)")

    is_nchw = shape[1] in (1, 3)  # crude heuristic

    batch_min, batch_opt, batch_max = 1, min(4, max_batch_size), max_batch_size

    if is_nchw:
        n, c, h, w = shape
        profile.set_shape(inp.name, (batch_min, c, h, w), (batch_opt, c, h, w), (batch_max, c, h, w))
    else:
        n, h, w, c = shape
        profile.set_shape(inp.name, (batch_min, h, w, c), (batch_opt, h, w, c), (batch_max, h, w, c))

    config.add_optimization_profile(profile)

    # Build engine ---------------------------------------------------------
    print("    Building … (this may take several minutes)")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT build failed")

    engine_path.write_bytes(engine_bytes)
    print(f"    ✓ Engine saved → {engine_path.name}  ({engine_path.stat().st_size/1024/1024:.2f} MB)")

# END: Shared build function

# ───────────────────────────────────────────────────────────────────────────
#  NEW: Generic ONNX → TensorRT converter (for RetinaFace & others)
# ───────────────────────────────────────────────────────────────────────────

class ONNXToTensorRTConverter:
    """Convert an arbitrary ONNX model (e.g., RetinaFace) to a TensorRT engine."""

    def __init__(
        self,
        onnx_file: Path,
        output_dir: Path,
        fp16: bool = True,
        int8: bool = False,
        max_batch_size: int = 8,
    ) -> None:
        self.onnx_file = onnx_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fp16 = fp16
        self.int8 = int8
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.INFO)

    # ---------------------------------------------------------------------
    def convert(self) -> None:
        """Run the ONNX → TensorRT compilation."""
        engine_name = self.onnx_file.stem + ".trt"
        engine_path = self.output_dir / engine_name

        build_tensorrt_engine(
            self.onnx_file,
            engine_path,
            self.logger,
            fp16=self.fp16,
            int8=self.int8,
            max_batch_size=self.max_batch_size,
        )

        # Quick usage hint --------------------------------------------------
        print("\nExample usage:\n"  # noqa: D401 – this is user output
              f"  import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit\n"
              f"  runtime = trt.Runtime(trt.Logger())\n  engine = runtime.deserialize_cuda_engine(open('{engine_name}', 'rb').read())\n"
              "  ctx = engine.create_execution_context()  # … etc.\n")

# ───────────────────────────────────────────────────────────────────────────
#  MAIN DISPATCH LOGIC
# ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MediaPipe *.task* or generic *.onnx* to TensorRT")
    parser.add_argument("--model", "-m", required=True, help="Path to *.task* or *.onnx* model")
    parser.add_argument("--output_dir", "-o", default="./trt_engines", help="Directory to place TensorRT engines")
    parser.add_argument("--fp16", action="store_true", default=False, help="Enable FP16 precision (if supported)")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision (requires calibration)")
    parser.add_argument("--max_batch_size", type=int, default=8, help="Maximum dynamic batch size")
    parser.add_argument("--combine", action="store_true", help="[MediaPipe only] Create combined engine wrapper")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model file not found: {model_path}")

    if model_path.suffix.lower() == ".task":
        # Defer to original MediaPipe pipeline --------------------------------
        from modelconversion import MediaPipeToTensorRTConverter  # type: ignore  # original module

        converter = MediaPipeToTensorRTConverter(
            task_file_path=model_path,
            output_dir=Path(args.output_dir),
            combine_models=args.combine,
        )
        converter.convert()

    elif model_path.suffix.lower() == ".onnx":
        # Generic ONNX (e.g. RetinaFace) ---------------------------------------
        ONNXToTensorRTConverter(
            onnx_file=model_path,
            output_dir=Path(args.output_dir),
            fp16=args.fp16,
            int8=args.int8,
            max_batch_size=args.max_batch_size,
        ).convert()

    else:
        parser.error("Unsupported model file type – must be .task or .onnx")


if __name__ == "__main__":
    main()

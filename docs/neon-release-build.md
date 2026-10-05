# Build the NEON single-camera release

For normal NEON use, download the complete tar.gz from the [GitHub Release](https://github.com/seamus-programming-stuff/neurogrip/releases/tag/v0.1.0-neon-jp512).
Git tracks application code, tests, model provenance and licenses. The large
ONNX graph, ARM64 wheels and complete offline archives are release assets.

Rebuilding requires an internet-connected development PC, Git and a local
Python environment with matching PyTorch/Torchvision. The original export
used PyTorch 2.11.0 and Torchvision 0.26.0. Install the appropriate official
PyTorch build for the PC before the export dependencies below. These commands
use Windows cmd from the repository root; do not run them on the NEON.

```cmd
py -m venv .venv
.venv\Scripts\python.exe -m pip install onnx==1.19.1 onnxruntime==1.23.2 huggingface-hub==2.1.1 opencv-python numpy
git clone https://github.com/DepthAnything/Depth-Anything-V2.git .vendor/depth-anything-v2
git -C .vendor/depth-anything-v2 checkout --detach a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
.venv\Scripts\python.exe build_neon_model.py
.venv\Scripts\python.exe -m pip download --dest jetpack_single\wheels --platform manylinux2014_aarch64 --python-version 38 --implementation cp --abi cp38 --only-binary=:all: --no-deps -r jetpack_single\requirements-offline.txt
.venv\Scripts\python.exe jetpack_single/app.py --self-test
.venv\Scripts\python.exe -m unittest discover -s tests
.venv\Scripts\python.exe package_neon_single.py
```

The exporter checks the upstream source revision, downloads the pinned official
metric Small checkpoint, precomputes its fixed portrait positional embedding,
exports IR8/opset14 and compares the ONNX output against the original model.
It writes the graph, license, attribution and model checksum into
`jetpack_single/models/`. The packager checks wheel hashes against
`jetpack_single/WHEEL_AUDIT.json`, builds both archives and verifies every file
by reading it back. Outputs are under `outputs/releases/`.

The bundled `VALIDATION.json` records verification of the original release.
Repeat target-version and device checks before publishing a modified release,
and update that report with the results. Package hashes cover the actual files
being packaged. Rebuilding is not guaranteed to produce bit-identical ONNX
bytes across different exporter versions or platforms.

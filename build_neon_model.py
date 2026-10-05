"""Reproducible PC export; not needed on the offline NEON."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / '.vendor' / 'depth-anything-v2'
REVISION = '3bc65d4e14a6786a61acec16453c50e12bf5f338'
CODE_REVISION = 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'
MODEL_ID = 'depth-anything/Depth-Anything-V2-Metric-Hypersim-Small'
FILENAME = 'depth_anything_v2_metric_hypersim_vits.pth'


def main():
    actual_revision = subprocess.check_output(
        ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_revision != CODE_REVISION:
        raise RuntimeError('Upstream V2 source must be pinned to ' + CODE_REVISION)
    from huggingface_hub import hf_hub_download
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    # Import the metric implementation, not the relative-depth implementation.
    sys.path.insert(0, str(SOURCE / 'metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2

    output = ROOT / 'jetpack_single' / 'models'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(hf_hub_download(MODEL_ID, FILENAME, revision=REVISION))
    print('Checkpoint downloaded; exporting portrait metric model.', flush=True)
    torch.set_num_threads(4)
    model = DepthAnythingV2(encoder='vits', features=64,
                           out_channels=[48, 96, 192, 384], max_depth=20)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    # Fixed 9:16, multiples of the 14px patch stride. All positional embedding
    # interpolation is constant folded, for TensorRT 8.5 / ORT 1.16 compatibility.
    generator = torch.Generator().manual_seed(42)
    example = torch.rand((1, 3, 448, 252), generator=generator)
    onnx_path = output / 'depth.onnx'
    started = time.monotonic()
    with torch.no_grad():
        expected = model(example).numpy()
        # TensorRT 8.5 does not implement ONNX cubic Resize. The positional
        # embedding depends only on our fixed input shape and trained weights.
        # Precompute it using the unchanged official function, then export it
        # as a buffer. The five depth-head linear Resize operations remain.
        tokens = torch.zeros((1, (448 // 14) * (252 // 14) + 1, 384))
        position = model.pretrained.interpolate_pos_encoding(tokens, 448, 252).detach()
        model.pretrained.register_buffer('neon_fixed_position', position)
        def fixed_position(module, x, w, h):
            return module.neon_fixed_position
        model.pretrained.interpolate_pos_encoding = MethodType(fixed_position, model.pretrained)
        if not np.allclose(model(example).numpy(), expected, atol=1e-6, rtol=1e-6):
            raise RuntimeError('Precomputed position changed model output')
        torch.onnx.export(model, example, str(onnx_path), opset_version=14,
                          dynamo=False, input_names=['input'], output_names=['depth'],
                          do_constant_folding=True)
    graph = onnx.load(onnx_path)
    # This graph uses only IR8 tensor/node features and opset14 operators.
    graph.ir_version = 8
    if any(a.s == b'cubic' for n in graph.graph.node if n.op_type == 'Resize'
           for a in n.attribute if a.name == 'mode'):
        raise RuntimeError('Cubic resize survived export')
    onnx.checker.check_model(graph)
    onnx.save(graph, onnx_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    actual = session.run(['depth'], {'input': example.numpy()})[0]
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=0.002, rtol=0.001):
        raise RuntimeError('ONNX numerical validation failed: %r' % error)
    manifest = {
        'schema_version': 1, 'model_id': MODEL_ID, 'revision': REVISION,
        'source_repo': 'https://github.com/DepthAnything/Depth-Anything-V2',
        'source_revision': CODE_REVISION, 'license': 'Apache-2.0',
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'sha256': hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        'input_name': 'input', 'output_name': 'depth',
        'input_shape': [1, 3, 448, 252], 'output_shape': [1, 448, 252],
        'input_size': 252, 'preprocessing': 'official_lower_bound',
        'units': 'metres', 'max_depth_m': 20.0,
        'opset': 14, 'onnx_ir_version': 8,
        'exporter': 'torch.onnx.export legacy; constant portrait shape; official positional embedding precomputed',
        'export_validation': {'max_absolute_error_m': error,
                              'torch_version': torch.__version__,
                              'onnxruntime_version': ort.__version__},
        'accuracy_note': 'Learned indoor metric estimates; no millimetre accuracy guarantee. '
                         '252px short side trades detail for embedded compute.'
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(SOURCE / 'LICENSE', output / 'LICENSE')
    (output / 'NOTICE.txt').write_text(
        'Depth Anything V2 Metric Hypersim Small, Apache License 2.0.\n'
        'Authors: Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao.\n'
        'Model: https://huggingface.co/' + MODEL_ID + '\n'
        'Code: https://github.com/DepthAnything/Depth-Anything-V2\n'
        'Modifications: exported metric inference to a fixed portrait ONNX graph; '
        'short side 252 pixels; no retraining.\n', encoding='utf-8')
    print(json.dumps({'model_bytes': onnx_path.stat().st_size,
                      'max_absolute_error_m': error,
                      'export_seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()

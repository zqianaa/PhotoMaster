# Recorded H20 inference environment

The October 13, 2026 one-image and five-image runs used an AutoDL Ubuntu 22.04 image with Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128 and CUDA 12.8. Project packages were installed in a virtual environment created with `--system-site-packages`, reusing the image's PyTorch. Both runs captured byte-identical `pip freeze` inventories.

- [constraints.txt](constraints.txt) pins the observed direct inference packages and selected support packages.
- [recorded-wheels.txt](recorded-wheels.txt) records the original official PyTorch, torchvision and FlashAttention wheel URLs including SHA-256 fragments.
- [observed-inventory.txt](observed-inventory.txt) records the wider image environment. It is evidence, **not an installable lock**: it includes unrelated image packages and omitted image-local build paths with unknown package versions.

## Recreate the tested image setup

Use the matching Linux x86_64 / Python 3.12 / PyTorch 2.8.0 / CUDA 12.8 image and run these commands from the repository root. The package choices and FlashAttention binary come from the verified run records. These commands consolidate the original setup into a portable recipe; this consolidated recipe has not been independently rerun from a new clean machine.

```bash
python3.12 -m venv --system-site-packages .venv-h20
source .venv-h20/bin/activate
python - <<'CHECK'
import sys, torch, torchvision
assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == '2.8.0+cu128'
assert torchvision.__version__ == '0.23.0+cu128'
assert torch.version.cuda == '12.8'
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
CHECK
python -m pip install -c environments/inference-h20-cu128/constraints.txt \
  -r requirements-inference.txt einops ninja
python -m pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb'
```

FlashAttention is installed separately after PyTorch. The recorded binary is specifically `cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64`; it cannot be substituted blindly across Python, PyTorch or C++ ABI versions.

For a clean virtual environment without the AutoDL image, the recorded base wheels are:

```bash
python -m pip install \
  'https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=4354fc05bb79b208d6995a04ca1ceef6a9547b1c4334435574353d381c55087c' \
  'https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=9cb3c13997afcb44057ca10d943c6c4cba3068afde0f370965abce9c89fcffa9'
```

Then install the project constraints and FlashAttention as above. This clean-environment alternative was not GPU-validated and still requires a compatible NVIDIA driver and Linux runtime; the recorded GPU run does not establish compatibility on other platforms.

## Exercise the CUDA extension before downloading weights

The repository preflight imports FlashAttention and its native CUDA extension but does not execute its kernels. This small check mirrors the original setup check and requires a GPU:

```bash
python - <<'CHECK'
import inspect, torch, flash_attn
from flash_attn import flash_attn_func
from diffusers import QwenImageTransformer2DModel
assert flash_attn.__version__ == '2.8.3'
assert 'zero_cond_t' in inspect.signature(QwenImageTransformer2DModel.__init__).parameters
q = torch.randn(1, 32, 4, 64, device='cuda', dtype=torch.bfloat16)
result = flash_attn_func(q, q, q)
torch.cuda.synchronize()
assert result.shape == q.shape
print('FlashAttention CUDA check passed:', tuple(result.shape))
CHECK
```

Return to the [quick start](../../docs/quickstart.md) for pinned model revisions and the single-GPU `FT_CKPT_IS_FULL=1` commands. Use a new `RUN_NAME` for each changed configuration. Diffusers 0.37.0 is required for the recorded 2511 `zero_cond_t` path; the older 0.36.0 implementation does not provide it.

The [execution evidence](../../docs/validation/h20-20261013/README.md) verifies the completed runs and documents source-provenance limits. These records establish one tested configuration, not a universal minimum GPU size, full dependency lock, deterministic regeneration guarantee or paper benchmark reproduction.

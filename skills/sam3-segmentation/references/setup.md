# First-machine setup

## What is and is not supplied

The skill includes the workflow, Python tools, a static viewer, configs and locked
Python environments. It does **not** include a photograph, private dataset,
model weights, GPU driver, provider account, existing inference service or API
key. Nothing requires the author's computer or a local endpoint.

| Path | What you need | Account / downloads |
|---|---|---|
| Demo, geometry, review, validation, export | Python 3.12/3.13; CPU | Small Python wheels; no account |
| SAM 3 on Linux x86-64 | Python 3.12, NVIDIA driver compatible with CUDA 12.8 wheels | Several GB of runtime/model downloads; gated Hugging Face access |
| SAM 3 on Linux ARM64 (`aarch64`, e.g. GB10 / DGX Spark) | Python 3.12, CUDA 13.0-compatible NVIDIA driver (R580 or newer) | CUDA 13.0 wheels selected automatically; model smoke check still required |
| Windows model inference | WSL2 Linux and an NVIDIA driver that supports WSL CUDA | Same model environment; do not install a second Linux kernel driver inside WSL |
| macOS/MPS | Base tools work; this backend has no MPS implementation | Use an approved Linux GPU host or implement and validate an adapter |
| CPU model inference | **Not implemented.** The tools refuse `--device cpu` | Use the base tools, or an approved Linux NVIDIA GPU machine |
| Hosted visual review | Optional official provider application/API | User's own permitted access and explicit photo-upload consent |

The portable wrapper is smoke-tested on an RTX 3090, not every GPU or OS.
The ARM64 CUDA 13.0 dependency path is configured but has not yet passed a
real-model smoke check here; dependency resolution alone is not inference validation.
The base suite was run on Linux with Python 3.12 and 3.13. Windows/macOS base
compatibility have not been tested here. CPU model inference is not implemented. Plan
for **12–24 GB GPU memory and 16–32 GB system RAM** as a conservative working
budget, not a measured minimum or guarantee. Batch size, crop sizes and concurrent
processes matter. Allow roughly **20–40 GB disk headroom** for environments,
weights and records; these are planning estimates, not a fixed download size.
High-resolution images can need substantially more RAM and raw-mask storage.
`plan` counts jobs and hypotheses, not guaranteed memory use or runtime.

## 1. Install basic tools

Install Git and `uv` using the official instructions appropriate for the OS:
- https://git-scm.com/downloads
- https://docs.astral.sh/uv/getting-started/installation/

Do not blindly copy `apt`, `pacman` or driver commands across machines. On
NixOS, use its declarative configuration or `nix shell`; on a managed machine,
ask its administrator. Do not pipe an uninspected installer into a privileged shell.

```sh
uv python install 3.12
sh "$S/scripts/run" base sync
sh "$S/scripts/run" base segment.py doctor
```

`S` is the absolute installed skill directory, **not** the repository root. An
installed copy includes scripts, references and the lock files for both
environments. Shell variables from one terminal/tool call may not persist in
another, so write the absolute path into each command.

`scripts/run` creates the environments **outside** the skill folder, in
`~/.local/share/sam3-segmentation/envs/` (or `$XDG_DATA_HOME/...`), one folder
per environment and lock-file checksum. Nothing is written into the skill
folder, so it can be read-only, replaced by an update or installed in several
places without rebuilding PyTorch. Set `SAM3_SEGMENTATION_HOME` to use another
writable folder. `sh "$S/scripts/run" model path` prints the model environment's
location; after an update changes a lock, delete the older folders there to
reclaim disk. The launcher is a POSIX shell script: on Windows, use it inside
WSL2 (the model needs WSL2 anyway) or Git Bash.

Optional base-only fallback without `uv` (not the locked inference path):

```sh
python3.12 -m venv .venv-segmentation
.venv-segmentation/bin/python -m pip install 'numpy>=1.26,<3' 'scipy>=1.15,<2' 'pillow>=11,<13' 'pycocotools==2.0.11' 'filelock>=3.19,<4'
.venv-segmentation/bin/python "$S/scripts/segment.py" demo --out /absolute/new-demo
```

Use this fallback for base tools only. `uv` is strongly recommended for the
pinned model environment. On Windows outside WSL, use the corresponding
`Scripts/python.exe` path for the base tools.

## 2. Check the GPU without changing it

```sh
uname -m
nvidia-smi
# Optional: shows a separately installed development toolkit, if any.
if command -v nvcc >/dev/null 2>&1; then nvcc --version; fi
```

Record the device index, driver and free memory. Never kill unrelated processes
to make space. Choose an available device explicitly with `--device cuda:N`.
`CUDA_VISIBLE_DEVICES` remaps indices; record that mapping if you use it. The
PyTorch wheel supplies CUDA userspace libraries, not the kernel driver. Install
or update drivers only through the machine's normal authorized process.

The `CUDA Version` in `nvidia-smi` describes the driver's supported CUDA level;
`nvcc --version` describes the installed toolkit. Neither selects PyTorch's
runtime. A CUDA 13-capable driver can run CUDA 12.8 wheels: matching the system
toolkit exactly is not required, and this workflow needs no separate toolkit.
The locked environment selects **CUDA 12.8 for Linux x86-64** and **CUDA 13.0 for
Linux ARM64**, not by parsing `nvidia-smi`. On GB10 / DGX Spark (`aarch64`), use
the normal `model` commands below; no local edits or global Torch install are
needed. GB10 uses shared system memory; if `nvidia-smi` reports memory usage as
`Not Supported`, inspect `free -h` too rather than treating it as zero available
memory. This is not a blanket compatibility claim for Jetson or other ARM GPUs.
See [NVIDIA's compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
and [PyTorch's published builds](https://pytorch.org/get-started/previous-versions/).

## 3. Create the isolated model environment

After download/trust approval:

```sh
sh "$S/scripts/run" model sync
sh "$S/scripts/run" model python - <<'PY'
import torch
from torchvision.ops import nms

print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable: check driver and environment"
torch.cuda.set_device("cuda:0")  # Choose an available device explicitly.
assert torch.cuda.is_bf16_supported(), "SAM 3 requires BF16 support"
print("GPU:", torch.cuda.get_device_name(), "capability:", torch.cuda.get_device_capability())
x = torch.eye(16, device="cuda:0", dtype=torch.bfloat16)
assert torch.equal(x @ x, x)
boxes = torch.tensor([[0., 0., 10., 10.]], device="cuda:0")
scores = torch.ones(1, device="cuda:0")
assert nms(boxes, scores, 0.5).tolist() == [0]
torch.cuda.synchronize()
print("CUDA tensor and Torchvision operator checks passed (no model weights loaded)")
PY
```

The source package is pinned to SAM 3 commit
`0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`; Torch is pinned to 2.10.0 and
Torchvision to 0.25.0 from matching explicit `cu128` (x86-64) or `cu130` (ARM64)
indexes. Python remains pinned to the 3.12 series. The lock pins the resolved
dependencies for both Linux architectures. Model
code is executable third-party code; a revision pin prevents silent drift, but
is not a security review. The base environment intentionally has no Torch.

The checks above exercise CUDA kernels without downloading weights; passing
them is not a SAM 3 test. Run the generated-shape model control below after
obtaining access. Changing the lock creates a new external environment; changing
Torch/CUDA/hardware invalidates inference caches, so use a new run directory.

## 4. Obtain gated weights legitimately

1. Visit https://huggingface.co/facebook/sam3 and read/accept the applicable terms.
2. Wait for access if necessary.
3. Authenticate locally; never paste a token into a conversation:

```sh
sh "$S/scripts/run" model hf auth login
```

The backend downloads `sam3.pt` at revision
`3c879f39826c281e95690f02c7821c4de09afae7`, not a moving `main`. The Hugging Face
cache is reused. Pass `--checkpoint /path/to/sam3.pt` to avoid a new download;
the actual file checksum, not its filename, identifies it. For offline use,
install dependencies and obtain weights on an authorized connected machine first.
Authentication, accepted terms and cache availability are separate checks.

## 5. Start tiny, then approve the full plan

First run an actual model control on generated shapes (weights/access required):

```sh
sh "$S/scripts/run" model model_smoke.py --out /absolute/new-model-check --device cuda:0
```

It checks real APIs and masks at 128×128, independent point prompts of different
lengths, batch/single disagreement and RLE round trips. It does not evaluate
natural-image accuracy. Inspect the minimum/individual IoUs, not just the median;
if it fails, keep its records and repeat in a new directory with `--batch-size 1`.

Then prepare an image and run `plan`. Start with one pass, no grid or a sparse grid,
and a small batch. The included `configs/smoke.json` is for checking API geometry,
not for high-quality annotation. After that works, inspect crops and authorize a
larger configuration. Loading a model is not proof its masks are semantically right.

## Problems and recovery

- **401/403/gated repository:** check account access and acceptance with the user.
  Do not try to bypass the gate or use somebody else's credentials.
- **CUDA unavailable / driver error:** inspect driver compatibility and the
  correct environment. Do not run the base environment for GPU commands.
- **Out of memory:** stop only this run, reduce `--batch-size` to 8 or 1 and use a
  new run if runtime identity changes. Do not resize the source secretly or kill
  other users' processes. Smaller crops can change the object scale; record them.
- **BF16 unsupported:** the pinned SAM 3 code casts some layers to BF16 itself, so
  FP32 is not an option (an FP32 run fails with a dtype error). Use a GPU with
  BF16 support (NVIDIA Ampere or newer). CPU inference is not implemented.
- **Missing SciPy/Pillow/pycocotools:** use the declared environment, not ad hoc
  installs in an unrelated global environment.
- **Unsupported platform / missing wheel:** the model lock targets Linux
  x86-64 and ARM64 only; Windows needs WSL2. On ARM64, confirm `uname -m` is
  `aarch64` and the driver supports CUDA 13.0. Do not substitute x86-64 wheels,
  assume all ARM/Jetson systems work, or silently fall back to CPU. Native
  compilation/toolchain work is outside this skill.
- **Interrupted inference:** rerun with the same canonical image, config,
  checkpoint, device, precision and batch size. Completed requests are cached.
  A different runtime belongs in a new run. Partial raw indexes are explicit.
- **Too many masks / implausibly few masks:** inspect `audit`, raw hypotheses,
  prompts, object scale and gate counts. Do not tune counts to a desired answer.
- **No outline / empty mask:** test on one easy object before blaming the dense
  image. A box, malformed SVG or all-zero polygon is not a usable mask.

## Optional automated viewer check

With permission for the additional browser download:

```sh
sh "$S/scripts/run" base --with playwright playwright install chromium
sh "$S/scripts/run" base --with playwright python "$S/scripts/browser_smoke.py" /path/run/export --out /absolute/new-browser-check
```

The check starts a loopback-only HTTP server and an isolated test browser, then
checks desktop/mobile, theme variables, selection, controls and downloads. It
never takes over the user's shared browser or profile. On machines where the
bundled browser lacks system libraries, use an appropriate dedicated browser via
`--browser-executable /path/to/test-browser`, installed through the OS's normal
package mechanism. Do not bypass a shared-browser launcher's profile safeguards.

## Licence and sharing boundary

This repository's MIT licence covers its own skill/tools only. SAM source,
weights, dependencies, photos, annotation rights and hosted-provider terms remain
separate. Review current upstream terms and acknowledge SAM where required. The
Hugging Face access gate is not removed by copying this skill. This is not legal
advice or a promise of unrestricted downstream training rights.

# AMD ROCm 和 Apple MPS

GAID 的 CLI 和工作台默认使用 `auto`。PyTorch 能访问 NVIDIA CUDA 或 AMD ROCm 时使用该设备，其次使用 Apple MPS，最后使用 CPU。CLI 可通过 `--device` 指定设备。

## Windows 11 上的 AMD ROCm

使用 AMD 官方 ROCm 7.2.1 的 Python 3.12 x64 wheel，包括 PyTorch `2.9.1+rocm7.2.1` 和 torchvision `0.24.1+rocm7.2.1`。显卡和驱动要求见 [AMD 的 Windows 支持表](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)。

1. 在项目根目录用 `py -3.12 -m venv .venv` 创建虚拟环境。
2. 执行 AMD 官方的两组安装命令，分别安装 ROCm 运行库和 PyTorch。将命令里的 `pip` 换成 `& .\.venv\Scripts\python.exe -m pip`，确保写入项目的虚拟环境。只需安装其中的 torch 和 torchvision。 [AMD Windows 安装说明](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html)
3. 安装 GAID 并确认 PyTorch 已识别 AMD GPU。

```powershell
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

版本输出应包含 `rocm`，最后一项应为 `True`。PyTorch 将 ROCm 设备表示为 `cuda`，因此可用 `--device cuda` 或 `--device cuda:0`。工作台仍可双击 `start-workbench.vbs` 启动。 [PyTorch HIP 设备说明](https://docs.pytorch.org/docs/2.11/notes/hip.html)

## Linux 上的 AMD ROCm

在 AMD 支持的 Linux x86-64 环境中安装相应的驱动，使用 PyTorch 发布的 ROCm 7.2 构建。

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/rocm7.2
./.venv/bin/python -m pip install -e .
./.venv/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

GAID 默认设置 `MIOPEN_FIND_MODE=FAST`，减少 ROCm 首次遇到新尺寸时的卷积算法搜索。启动前设置的值优先。

## Apple Silicon 上的 MPS

使用 Apple Silicon、macOS 14 或更新版本及原生 arm64 Python 3.12。安装 Xcode 命令行工具后，使用 PyPI 的 PyTorch wheel。 [Apple 的 PyTorch MPS 安装说明](https://developer.apple.com/metal/pytorch/)

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0
./.venv/bin/python -m pip install -e .
./.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
./.venv/bin/gaid process ./image.png --out ./outputs/whole --whole --device mps
```

MPS 加速 L0 和超分模型。L0 使用单精度 FFT，显存不足时在 CPU 上重试。部位分区与 SAM2 在 CPU 上运行。要在 Linux 或 macOS 使用工作台，在项目根目录准备模型后运行以下命令。

```bash
./.venv/bin/python -m tools.assets fetch
./.venv/bin/gaid ui
```

ROCm 和 MPS 路径已做静态和模拟验证，实际速度与显存占用需以相应设备为准。

# GPT AnimeImage Denoise

GPT AnimeImage Denoise（GAID）在本机清理二次元插画的鱼鳞纹和重复颗粒。可以在网页工作台对比和精修单张图，用命令行批量处理，也可以交给 Codex 或 Claude Code 调用。

[安装](#安装) · [工作台](#工作台) · [命令行](#命令行) · [AI Agent](#ai-agent) · [模型](#使用的模型)

## 安装

在 Windows 安装 Python 3.12，下载项目源码并解压，或克隆仓库。在包含 `pyproject.toml` 的目录打开 PowerShell。

以下命令安装 NVIDIA CUDA 12.8 版 PyTorch。使用 CPU 时，把下载地址末尾的 `cu128` 改成 `cpu`。 [PyTorch 版本说明](https://pytorch.org/get-started/previous-versions/#v2110)

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -e .
```

AMD 显卡和 Apple Silicon 的安装步骤、驱动要求及验证命令见 [AMD ROCm 和 Apple MPS](docs/accelerators.md)。

### 检查安装

```powershell
.\.venv\Scripts\gaid.exe doctor
.\.venv\Scripts\gaid.exe doctor --whole
```

`doctor` 显示实际使用的解释器、PyTorch 构建、推理设备和模型目录，并给出修复方法。只用整图纯 L0 时加 `--whole`。如果移动目录后 `gaid.exe` 已无法启动，可以从源码运行诊断。 [环境诊断](docs/cli.md#环境诊断)

### 切换 CPU 或 CUDA 版 PyTorch

已有安装时，先退出工作台和使用同一 `.venv` 的 GAID 进程。在项目根目录运行下面的命令切到 NVIDIA CUDA 12.8 版。要切到 CPU 版，将最后的 `cuda` 改为 `cpu`。

```powershell
& .\.venv\Scripts\python.exe -X utf8 .\tools\switch_pytorch.py cuda
```

脚本会先下载适用于 CPU 或 CUDA 的 `torch` 和 `torchvision` wheel。两个都下载成功后才从本地替换，下载失败时保留现有安装。完成后会实际导入这两个包，显示构建版本和 GAID 在 `auto` 模式下可用的设备。

### 首次出图

把 `image.png` 放到项目根目录，运行整图纯 L0。`--whole` 跳过部位分区，此命令无需模型权重。

```powershell
.\.venv\Scripts\gaid.exe process .\image.png --out .\outputs\whole --whole
```

成品 `result.png` 位于 `outputs/whole/` 下的案例子目录。`process` 和 `batch` 输出的成品保留定向后的原尺寸、Alpha 和 ICC。

### 准备分区模型

用「按部位精修」或按人物部位运行命令行时，需要准备约 3.3 GB 的分区模型。默认保存在项目的 `models/`，已有模型可以复用。 [模型准备](docs/models.md)

```powershell
& .\.venv\Scripts\python.exe -m tools.assets fetch
```

## 使用

### 工作台

完成安装后，双击 [start-workbench.vbs](start-workbench.vbs)，浏览器会打开本地工作台。环境缺少依赖时会弹出缺失项和修复步骤。需要查看终端日志时运行 [start-workbench.cmd](start-workbench.cmd)。

1. 打开、拖入或粘贴图片，工作台会在后台出图。缺少分区模型时，新图进入「整图降噪」，用默认纯 L0 即可出图。有分区模型时，新图进入「按部位精修」，默认处理皮肤、面部和头发。可在「设置 → 新图默认值」指定模式，或设为「每次选择」。
2. 在「成品检查」拖动分割线查看差异，也可以把当前成品固定为对照 A。
3. 整图模式点「降噪效果」调参。按部位模式点「编辑流程」，用画笔、套索或 SAM2 修改选区。两个模式均可导出原尺寸 PNG。

[工作台手册](docs/workbench.md)

### 命令行

准备好分区模型后，可按人物部位处理单张图或整个目录。`batch --directory` 会递归扫描输入目录，`--out` 要放在输入目录之外。

```powershell
.\.venv\Scripts\gaid.exe process .\image.png --out .\outputs\single
.\.venv\Scripts\gaid.exe batch --directory .\images --out .\outputs\batch
```

`process` 和 `batch` 都接受 `--whole`。案例目录包含成品 `result.png`、权重图 `weight.png` 和参数报告 `report.json`。保留同级 `.cache` 后，可以用 `preview` 调整混合强度。 [命令行手册](docs/cli.md)

```powershell
.\.venv\Scripts\gaid.exe --version
.\.venv\Scripts\gaid.exe process --help
```

### AI Agent

安装 GAID 后，在仓库根目录运行一次用户级 skill 安装。只需安装一个客户端时，在命令末尾加 `--agent codex` 或 `--agent claude`。

```powershell
& .\.venv\Scripts\python.exe -X utf8 .\tools\install_agent_skill.py
```

| 客户端 | 调用方式 | 用户级 skill 目录 |
| --- | --- | --- |
| Codex | `$gpt-animeimage-denoise` | `~/.agents/skills/` |
| Claude Code | `/gpt-animeimage-denoise` | `~/.claude/skills/` |

Claude Code 设置了 `CLAUDE_CONFIG_DIR` 时，安装器使用该目录下的 `skills/`。

在任意本地工作目录打开 Agent，给它原图路径和处理要求。也可以用当前会话中生图工具返回的原图。例如「跳过部位分区，给 `D:\images\插画.png` 整图去噪，交付原尺寸 PNG」。成品默认写入任务目录下的 `outputs/agent-denoise/`。

安装器会记录 GAID、Python 和模型的路径。更新 skill 或更换环境和模型目录后，重新运行安装命令。移动项目目录后，先在新位置重建 Python 环境，再安装 skill。 [Skill 说明](.agents/skills/gpt-animeimage-denoise/SKILL.md) · [复用本机模型](docs/models.md#复用本机模型)

## 使用的模型

| 模型 | 用途 | 上游许可与说明 |
| --- | --- | --- |
| [SkyTNT ISNet](https://github.com/SkyTNT/anime-segmentation) | 人物与背景分割 | 源码 Apache-2.0，权重未单列许可 |
| [AnimeSeg Mask2Former v4](https://huggingface.co/suzukimain/AnimeSeg) | 人物部位分割 | 未声明通用许可证 |
| [See-through](https://huggingface.co/24yearsold/l2d_sam_iter2#licence) | 辅助识别部位和外物 | Apache-2.0 |
| [SAM 2.1 Hiera Large](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md#license) | 修正轮廓和点选外物 | Apache-2.0 |
| [Real-ESRGAN x4plus Anime 6B](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4) | 可选降噪模型 | [源码 BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) |
| [APISR 2× RRDB](https://github.com/Kiteretsu77/APISR#license) | 可选降噪模型 | 源码 GPL-3.0，作者注明学术用途，[权重镜像](https://huggingface.co/camenduru/APISR)未声明许可 |

## 开发

测试、wheel 构建和调试工具见[开发说明](docs/development.md)。编辑 `.agents/skills/gpt-animeimage-denoise/SKILL.md` 后，从项目根目录同步 Claude Code 的项目入口。

```powershell
Copy-Item -LiteralPath '.agents/skills/gpt-animeimage-denoise/SKILL.md' -Destination '.claude/skills/gpt-animeimage-denoise/SKILL.md'
```

## 致谢与许可证

项目源码采用 [MIT 许可证](LICENSE)，其中收录 KONOK-A 和 CXP-2024 的版权声明。模型的许可信息见上表和各自的上游页面。

整图降噪方案参考了 [CXP-2024/gpt-anime-denoise](https://github.com/CXP-2024/gpt-anime-denoise)。

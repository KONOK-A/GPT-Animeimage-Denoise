# GPT AnimeImage Denoise

在本机清理二次元插画的鱼鳞纹。工作台会自动识别人物部位，默认只对皮肤、面部和头发降噪。可以用画笔或套索修改选区，调节局部强度，对比原图后导出原尺寸 PNG。

## 安装

在 Windows 安装 Python 3.12，把仓库下载到本机，再在项目根目录打开 PowerShell。下面以 CUDA 12.8 版 PyTorch 为例。使用 CPU 时，把下载地址末尾的 `cu128` 改成 `cpu`。 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v2110)

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -e .
```

## 使用

**工作台和人物分区**需要先准备模型，首次下载约 3.3 GB。已有模型可直接复用。 [模型准备](docs/models.md)

```powershell
& .\.venv\Scripts\python.exe -m tools.assets fetch
```

在 Windows 双击 [start-workbench.vbs](start-workbench.vbs) 即可在后台打开工作台。需要查看终端日志时，使用 [start-workbench.cmd](start-workbench.cmd)。

把图片拖进浏览器里的工作台。默认成品会在后台生成，可在「成品检查」对比和导出。需要精修时，进入「编辑流程」调整部位分区、处理范围或降噪效果。 [工作台手册](docs/workbench.md)

**命令行**支持单图和批量去噪。按人物部位去噪用 `gaid process`。想跳过模型下载，可以直接运行整图纯 L0。

```powershell
# 单张分区去噪
.\.venv\Scripts\gaid.exe process .\image.png --out .\outputs\single

# 整图纯 L0
.\.venv\Scripts\gaid-whole.exe .\image.png --out .\outputs --method l0_0.0005
```

[命令行手册](docs/cli.md) · [开发说明](docs/development.md)

### 在任意项目用 AI Agent 去噪

按上文在本机安装 GAID 并准备分区模型，在 GAID 仓库运行一次用户级安装。

```powershell
& .\.venv\Scripts\python.exe -X utf8 .\tools\install_agent_skill.py
```

安装后，从任意本地工作目录启动 Agent，即可调用同一份降噪 skill。

- **Codex** 从 `~/.agents/skills/` 发现用户级 skill。输入 `$gpt-animeimage-denoise`。
- **Claude Code** 从 `~/.claude/skills/` 发现用户级 skill。输入 `/gpt-animeimage-denoise`。若设有 `CLAUDE_CONFIG_DIR`，安装器会使用该目录下的 `skills/`。

调用时一起给出原图和处理要求，例如「给 `D:\images\插画.png` 去鱼鳞纹，尽量保留发丝和背景，交付原尺寸 PNG」。也可以使用同一会话中生图工具返回的原图。默认结果位于调用时的工作目录下的 `outputs/agent-denoise/`。

在安装命令末尾加 `--agent codex` 或 `--agent claude` 可选择客户端。安装会记录 GAID 路径、所用的 Python 和模型目录，支持 [复用本机模型](docs/models.md#复用本机模型)。移动项目、更换环境或更新 skill 后，重新运行安装命令。

维护时以 [`.agents` 中的 SKILL.md](.agents/skills/gpt-animeimage-denoise/SKILL.md) 为源文件。更新后在项目根目录同步 [Claude Code 项目入口](.claude/skills/gpt-animeimage-denoise/SKILL.md)。

```powershell
Copy-Item -LiteralPath '.agents/skills/gpt-animeimage-denoise/SKILL.md' -Destination '.claude/skills/gpt-animeimage-denoise/SKILL.md'
```

## 使用的模型

| 模型 | 用途 | 许可 |
| --- | --- | --- |
| [SkyTNT ISNet](https://github.com/SkyTNT/anime-segmentation) | 人物与背景分割 | 源码 Apache-2.0，权重未单列许可 |
| [AnimeSeg Mask2Former v4](https://huggingface.co/suzukimain/AnimeSeg) | 人物部位分割 | 未声明通用许可证 |
| [See-through](https://huggingface.co/24yearsold/l2d_sam_iter2#licence) | 辅助识别部位和外物 | Apache-2.0 |
| [SAM 2.1 Hiera Large](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md#license) | 修正轮廓和点选外物 | Apache-2.0 |
| [Real-ESRGAN x4plus Anime 6B](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4) | 可选降噪模型 | [源码 BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) |
| [APISR 2× RRDB](https://github.com/Kiteretsu77/APISR#license) | 可选降噪模型 | 源码 GPL-3.0，作者注明学术用途，[权重镜像](https://huggingface.co/camenduru/APISR)未声明许可 |

## 致谢与许可证

整图降噪方案参考了 [CXP-2024/gpt-anime-denoise](https://github.com/CXP-2024/gpt-anime-denoise)。

Copyright © 2026 KONOK-A · [MIT](LICENSE)

---
name: gpt-animeimage-denoise
description: Use the local GPT AnimeImage Denoise tool to reduce fish-scale artifacts and repetitive grain patterns in GPT-generated illustrations. Compare the result with the source and export a PNG at the original pixel dimensions. Use when a user asks to denoise an existing illustration or explicitly requests denoising after generating an image with GPT.
---

# GPT AnimeImage Denoise

## 取得本次要处理的原图

- 使用用户指定的本机图片，或本次生图任务返回的原始文件。只有聊天预览时，先用生图工具提供的原文件路径或导出功能。无法读取原文件时，请用户提供图片文件。
- GPT Image Playground 的 `generate --json` 返回 `tasks[].files`。只有任务 ID 或图片 ID 时，用 `gpt-image-playground image <ID> --json` 解析 `files`，按指定 ID 对应到原图。Windows 上如果 PATH 指向应用本体，调用安装目录里的 `bin\gpt-image-playground.exe`。应用数据目录内的图片按只读处理。
- 按用户选中的文件逐张处理。用户指定整个目录时才使用 `gaid batch --directory`，它会递归扫描。把 `--out` 放在输入目录之外，并逐张检查 `summary.csv` 或 `records.json`。

## 定位本机 GAID

用户级安装时，读取已加载的 `SKILL.md` 同目录下的 `gaid-installation.json`。启动命令用其中的绝对路径 `python_executable`，把 `model_dir` 作为 GAID 子进程的 `GPT_ANIMEIMAGE_DENOISE_MODEL_DIR`，并到 `project_root` 查阅项目文档。从当前任务目录解析输入和输出路径。

使用源码仓库内的入口时，GAID 仓库是 `SKILL.md` 所在目录向上的第三级目录。使用这个仓库的 `.venv` 解释器和 `models/`，或机器上已配置的 `GPT_ANIMEIMAGE_DENOISE_MODEL_DIR`。用户级安装路径移动或失效时，用有效的 GAID 环境重新运行该仓库的 `tools/install_agent_skill.py`。

## 默认按部位去噪

遇到模型缺失或校验失败时，查阅 GAID 仓库的 `docs/models.md`。分区模型完整下载约 3.3 GB。

默认纯 L0（`0.0005`）处理皮肤、面部和头发，保留选区外的像素。按用户指定的输出位置保存，每次处理使用独立目录，原图保持原位。以下 PowerShell 7 示例通过已定位的 Python 以 UTF-8 调用 `gaid`，在当前任务目录的 `outputs/agent-denoise/` 下创建案例。将示例中的三个路径替换为实际值。

```powershell
$python = '<已定位的 GAID Python 绝对路径>'
$gaidModels = '<已定位的模型目录绝对路径>'
$gaidArgs = @('-X', 'utf8', '-m', 'gpt_animeimage_denoise.cli')
$source = (Resolve-Path -LiteralPath 'D:\images\原图.png').Path
$runDir = Join-Path (Get-Location).Path ('outputs\agent-denoise\' + [guid]::NewGuid().ToString('N'))
$previousModelDir = $env:GPT_ANIMEIMAGE_DENOISE_MODEL_DIR
try {
    $env:GPT_ANIMEIMAGE_DENOISE_MODEL_DIR = $gaidModels
    $log = & $python @gaidArgs process $source --out $runDir
    $gaidExit = $LASTEXITCODE
} finally {
    $env:GPT_ANIMEIMAGE_DENOISE_MODEL_DIR = $previousModelDir
}
if ($gaidExit -ne 0) { throw 'gaid process failed. See stderr.' }
$case = $log | Select-Object -Last 1 | ConvertFrom-Json -ErrorAction Stop
if ($case.status -ne 'complete') { throw "Unexpected GAID status: $($case.status)" }
$result = Join-Path $case.output 'result.png'
```

批量处理也用同一个解释器和模型路径。单图命令结束后，读取最后一行 JSON 的 `output` 取得案例目录。`result.png` 是全尺寸成品，`original.png` 是同方向的原图。`weight.png` 越亮表示降噪权重越高，黑色处保持原图。`report.json` 记录输入、参数和状态。报错时查看 stderr，遇到 CUDA 不可用或显存不足可以改用 `--device cpu`。

## 核对、微调、交付

打开成品，核对宽高与定向后的原图和报告 `size` 一致。源图带有 Alpha 或 ICC 色彩配置时一并核对。逐张在原尺寸下比较 `original.png` 和 `result.png` 的同坐标裁片，既看平坦色面上的鱼鳞纹是否减轻，也看发丝、线稿、高光和渐变是否保真。局部试算时给裁片四周留出上下文，比较中央区域，再回到整图成品复核。

### 修正处理范围

- 在成品上找到残纹，再对照 `weight.png` 或工作台「实际处理」查看同一位置。处理头发时，沿发顶、刘海、鬓发查到肩后和发梢。漏选处先补选，窄发束只有低权重过渡时检查局部羽化。`report.json` 中的 `metrics.active_percent` 和 `metrics.changed_pixels` 可辅助排查覆盖情况。
- 以原图辨认材质。归属有误时在「部位分区」修正，只需增减处理面积时在「处理范围」画入或移出。小块用画笔或套索。交叠轮廓可用 SAM2，在目标内部放目标点，在相邻衣料或道具上放排除点。沿原图边界检查预览，留意指缝和袖口，贴合后应用，越界处用套索收边。编辑流程见 GAID 仓库的 `docs/workbench.md`。
- 用户要求保留原像素的织纹、刺绣和饰边应落在零权重区。比较原图与导出 PNG 的 RGB，把发生变化的像素标回原图，检查纹样内部及其与头发、皮肤的交界。移出误选的衣料后，再核对保护处与原图逐像素一致。

### 调整候选和混合强度

- 对照同一位置的「全图降噪预览」与成品。预览显示渐变恢复前的候选。候选干净、成品仍有纹时，先查看该部位的实际处理权重，低权重会保留更多原图纹理。固定权重，将当前成品与渐变恢复设为 `0` 的成品对照，同时观察残纹和原有的宽缓明暗变化。
- 自动选区远离边缘和分区交界的核心可用整体强度乘部位强度估算权重。羽化和线条保护会降低权重，分区交界会平滑过渡。手动补入未启用的部位时，该部位的强度因子按 `1` 计，需复查与周围的过渡。
- 全图预览也有残纹时，在同一块原图上试调 L0 或比较其他算法。
- 工作台的算法设置作用于整张图。若某一部位需要不同的算法或 L0 强度，从同一原图分别生成候选并恢复渐变，再按各自核对过的软权重合成。脚本可用 `WorkbenchSession.candidate()` 取得混合前的浮点候选。给 `gradient_preserving_candidate` 显式传入各候选对应的 `restoration_gain`（工作台内置值为 `0.25`），以及按 `界面 sigma × min(宽, 高) / 2048` 换算成原图像素的 `sigma`，然后用 `blend` 合成。接口见 GAID 仓库的 `src/gpt_animeimage_denoise/workbench/session.py` 和 `src/gpt_animeimage_denoise/filters.py`。
- 减轻效果时，保留案例和旁边的 `.cache`，用 `preview` 复用分区和候选图。例如 `& $python @gaidArgs preview $case.output --out (Join-Path $runDir 'lighter.png') --strength 0.7`。自由调整强度的基础案例需以 `--strength 1` 处理。更换算法或 L0 时重新生成候选。复用同源二值校正用 `gaid process --edits`，复现羽化、局部原图比例和分部位强度时保留工作台草稿或对应的软权重与设置。应用校正见 GAID 仓库的 `docs/cli.md`。

### 局部返修与交付

用户只要求返修某个部位时，保留上一版成品，并按以下步骤生成局部更新。

1. 从同一张定向原图生成新候选并恢复渐变，用 `repaired = blend(original, restored_candidate, target_weight)` 计算返修成品。`target_weight` 是调整后的降噪权重。要恢复原图的像素，将该权重设为 `0`。
2. 另建返修蒙版 `repair_mask`，返修区核心设为 `1`，范围外设为 `0`，边缘按需过渡。用 `blend(previous, repaired.astype(np.float32) / 255, repair_mask)` 合回上一版成品，另存原尺寸 PNG 并保留源图的 Alpha 和 ICC。
3. 比较返修区域与上一版的同位置裁片，并逐像素验证返修蒙版为零处与上一版一致。首次降噪时，以定向后的原图核对降噪权重为零的像素。

命令行处理或预览时核对单图状态 `complete`，工作台导出时确认当前成品 B 已更新。给出选定的原尺寸 PNG 路径或可点击链接，说明处理的部位和采用的设置。若只能检查文件和权重，或某处仍有明显纹理，说明实际核对范围和剩余问题。默认保留源文件和案例，用户需要时再附报告和权重图。

用户明确希望整图平滑时，先检查源图的 EXIF 方向。带旋转或镜像方向标记时，用 `PIL.ImageOps.exif_transpose` 将方向写入像素并另存副本，再把副本作为输入。用同一个解释器调用整图入口 `& $python -X utf8 -m denoise $source --out $runDir --method l0_0.0005`，无需分区和超分模型。输出 `<输入文件名去掉扩展名>_l0_0.0005.png` 后，核对宽高与定向后的原图一致。此方法输出 RGB，去掉透明通道和 ICC。整图处理仍需保留这些信息时，在工作台手动设定处理范围并导出。其他参数见 GAID 仓库的 `docs/cli.md`。

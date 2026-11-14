# 命令行

先按 [README](../README.md#安装) 安装项目。以下命令在项目根目录的 PowerShell 中运行。

## 按人物部位去噪

先 [准备分区模型](models.md)。

```powershell
.\.venv\Scripts\gaid.exe process .\image.png --out .\outputs\single
.\.venv\Scripts\gaid.exe batch --directory .\images --out .\outputs\batch --method esrgan-l0
```

默认处理皮肤、面部和头发。每张图在输出目录下有独立子目录，其中 `result.png` 是成品，`weight.png` 是处理权重，`report.json` 记录参数和输入信息。`batch --directory` 会递归扫描原图，跳过文件名标有 `denoise` 等处理后缀的图片。输出目录须位于输入目录之外，批量结果可在 `summary.csv` 中查看。

### 常用参数

下表为默认 `fusion` 模式的设置。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--method` | `l0` | 还可选 `esrgan`、`esrgan-l0` 或 `apisr` |
| `--device` | `auto` | 有 CUDA 时用 GPU，也可指定 `cpu`、`cuda` 或 `cuda:N` |
| `--strength` | `1` | 整体混合强度，范围 0–1 |
| `--l0` | `0.0005` | L0 平滑强度，调大会压掉更多纹理 |
| `--downscale` | `0.6` | ESRGAN 推理前的缩放比例，仅用于含 ESRGAN 的方案 |
| `--structure-protection` | `0.2` | 线条保护比例，范围 0–1 |
| `--restoration-gain` | `0.25` | 渐变恢复比例，范围 0–1 |
| `--sigma` | `12` | 渐变恢复的滤波尺度，以短边 2048 像素为基准 |

### 应用校正

将工作台导出的 `corrections.json` 与蒙版一起保留。给 `process` 加 `--edits .\corrections\corrections.json`，即可对同一源图应用校正。源图文件的 SHA-256 和定向后的 `[宽, 高]` 须与 JSON 匹配，蒙版须与定向后的源图同尺寸。也可参照 [校正模板](../examples/correction.template.json) 自行填写。

批量使用 `--edits-dir`，每份校正 JSON 需填写对应案例的 `id`。程序按 `id` 配对并核对源图哈希。可用 `batch --manifest` 复用由 `gaid inventory` 生成的 `manifest.json`，固定批次的图片清单和 ID。

## 整图去噪

```powershell
.\.venv\Scripts\gaid-whole.exe .\image.png --out .\outputs\whole --method l0_0.0005 --device cpu
```

整图模式无需分区模型。默认 `esrgan60_l0` 将图缩到 60%，用 Real-ESRGAN 放大后回到原尺寸，再做 L0 平滑。`--all` 会运行全部五种整图方案，并保存一张对比图。纯 L0 可选 `l0_0.0005` 或 `l0_0.001`。

## 从缓存预览

用 `process` 完成一张图后，将命令输出中的案例子目录传给 `preview`，并保留旁边的 `.cache`。

```powershell
.\.venv\Scripts\gaid.exe preview .\outputs\single\image-0123456789 --out .\outputs\preview.png --strength 0.7
```

预览在 CPU 上复用已保存的选区和浮点候选。可调 `--strength`、`--restoration-gain` 和 `--sigma`，省略时沿用案例参数。要自由调整整体强度，先以 `--strength 1` 生成案例。每次预览指定新的 PNG 路径。更换算法、L0 或选区需重新处理。

在相应命令后加 `--help` 可查看完整参数。

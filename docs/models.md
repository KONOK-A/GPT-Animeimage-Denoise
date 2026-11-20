# 模型准备

按部位精修和分区去噪需要约 3.3 GB 的模型文件。工作台的「整图降噪」和命令行的 `--whole` 可用纯 L0 直接出图。Real-ESRGAN 和 APISR 的权重各约 18 MB，首次选用对应方案时自动下载。

## 下载与检查

完成 [安装](../README.md#安装) 后，在项目根目录运行。`fetch` 会复用已通过校验的文件，补齐缺少的分区模型。

```powershell
& .\.venv\Scripts\python.exe -m tools.assets fetch
& .\.venv\Scripts\python.exe -m tools.assets verify
```

默认目录为 `models/`。下载来源、相对路径和 SHA-256 见 [资产清单](../src/gpt_animeimage_denoise/model_manifest.json)。

如果要同时检查 Python 和推理设备，可用 `gaid doctor --verify-models`。 [环境诊断](cli.md#环境诊断)

## 复用本机模型

将示例路径换成已有的模型目录。以下命令直接使用该目录，当前 PowerShell 会话中启动的工作台和命令行均会继承此设置。

```powershell
$env:GPT_ANIMEIMAGE_DENOISE_MODEL_DIR = 'D:\existing-install\models'
& .\.venv\Scripts\python.exe -m tools.assets verify
```

要在当前项目保留独立副本，用 `import`。已有同名文件的 SHA-256 不一致时，命令会报错并保留原文件。

```powershell
& .\.venv\Scripts\python.exe -m tools.assets import 'D:\existing-install\models' --dest models
```

各模型的用途和许可见 [README](../README.md#使用的模型)。

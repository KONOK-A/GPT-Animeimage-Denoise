# 批量降噪对比

在相同人物分区和权重下对比纯 L0 与 Real-ESRGAN＋L0。页面可查看分区、实际权重、全图候选和最终合成结果。

推理需要 CUDA 和[本地模型](../../docs/models.md)。将样图直接放在 `inputs/` 下，在仓库根目录用 PowerShell 执行。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m tools.batch_acceptance.run_comparison --input .\inputs --out .\outputs\acceptance
& $py -m tools.batch_acceptance.build_review --run .\outputs\acceptance
Copy-Item .\tools\batch_acceptance\web\* .\outputs\acceptance\
```

打开 `outputs/acceptance/index.html`。审阅意见可导出为 JSON。重新导出页面时需保留原图、`native/`（含 `.cache/`）、`trace/` 和批次清单。

默认 `--l0 0.0005`、`--downscale 0.6`。运行 `& $py -m tools.batch_acceptance.run_comparison --help` 查看参数。换样图、参数、代码或模型后，请使用新的输出目录。

## 追加样图

完成并导出原批次后，可用相同代码、配置和模型追加样图。新结果写到 `extensions/`，最后将新案例并入原网页。

```powershell
& $py -m tools.batch_acceptance.append_comparison --parent .\outputs\acceptance --name additional --inputs .\more\sample.png
& $py -m tools.batch_acceptance.build_review --run .\outputs\acceptance\extensions\additional
& $py -m tools.batch_acceptance.merge_reviews --parent .\outputs\acceptance --extension .\outputs\acceptance\extensions\additional
```

## 比较线条保护与渐变恢复

从网页清单 `manifest.json` 选择一至三个案例 ID，复用已缓存的浮点候选生成对照图。

```powershell
& $py -m tools.batch_acceptance.denoise_ablation --run .\outputs\acceptance --out .\outputs\ablation --ids batch_0001
```

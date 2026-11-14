# 分区对比工具

用静态网页对照 See-through、Mask2Former 和实际的融合分区。可以分部位查看，并切换原图与白底输入、检查外物与待确认区域。导出页面时复用推理缓存。

## 对比模型与融合结果

在仓库根目录用 PowerShell 7 执行。将样图放入 `inputs/`，并按[模型说明](../../docs/models.md)准备模型。完整对比只接受合格的开发样本（`status=eligible`、`split=tune`）。

```powershell
$gaid = '.\.venv\Scripts\gaid.exe'
$py = '.\.venv\Scripts\python.exe'
& $gaid inventory .\inputs --out .\outputs\inventory
$inventory = Get-Content -Raw -LiteralPath .\outputs\inventory\manifest.json | ConvertFrom-Json
$items = @($inventory.items | Where-Object { $_.status -eq 'eligible' -and $_.split -eq 'tune' })
@{ items = $items } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath .\outputs\development-manifest.json -Encoding utf8
& $gaid batch --manifest .\outputs\development-manifest.json --out .\outputs\native --profile fusion
Copy-Item -LiteralPath .\outputs\development-manifest.json -Destination .\outputs\native\input-manifest.json
& $py -m tools.segmentation_review.build_full_comparison --native .\outputs\native --out .\outputs\segmentation-review
Copy-Item .\tools\segmentation_review\full_web\* .\outputs\segmentation-review\
```

打开 `outputs/segmentation-review/index.html`。保留 `native/.cache/` 和原图，以便重新导出。浏览器里的切换只影响显示，可导出审阅意见。

## 只对比两套模型

将源图路径、完整 SHA-256 和对应的双视图 `evidence-*.npz` 写入 `request.json`，路径相对于仓库根目录。

```json
{
  "cases": [
    {
      "id": "sample",
      "path": "inputs/sample.png",
      "sha256": "<source-sha256>",
      "evidence_cache": "outputs/native/.cache/evidence-<cache-id>.npz"
    }
  ]
}
```

可选字段 `see_original_cache` 指向另行保存的 See-through 原图分数。没有这个文件时，该页面只显示 See-through 的白底结果。

```powershell
& .\.venv\Scripts\python.exe -m tools.segmentation_review.build_assets --request .\request.json --out .\outputs\model-review
Copy-Item .\tools\segmentation_review\web\* .\outputs\model-review\
```

打开 `outputs/model-review/index.html`。

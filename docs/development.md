# 开发说明

按 [README](../README.md#安装) 安装。在项目根目录运行以下命令。Python 单元测试无需下载分区模型，前端测试使用 Node.js 24。

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests
node --test 'tests/web/*.cjs'
```

## 源码位置

- [`src/gpt_animeimage_denoise/`](../src/gpt_animeimage_denoise) 包含人物分区、降噪和命令行入口。
- [`workbench/`](../src/gpt_animeimage_denoise/workbench) 包含本地服务，`web/` 为浏览器界面。
- [`src/denoise.py`](../src/denoise.py) 是整图去噪入口。
- [`tests/`](../tests) 包含 Python 和前端测试。

## 构建 wheel

```powershell
& .\.venv\Scripts\python.exe -m pip wheel --no-deps --wheel-dir dist .
```

wheel 提供 `gaid`、`gaid-whole` 和 `gaid-eval`。准备模型的命令在源码仓库执行。

## 调试图像结果

- [分区对比](../tools/segmentation_review/README.md) 查看模型输出与最终分区。
- [批量对比](../tools/batch_acceptance/README.md) 检查成品、降噪候选与实际权重。

# HamiVisualizer

当前发布说明与完整使用文档请阅读 [README-v0.3.md](README-v0.3.md)。该文件虽然保留历史文件名，内容已更新为 HamiVisualizer 0.4 的准确说明。

## 快速开始

项目使用 Python 3.12、PySide6 6.6.x、NumPy 1.x 和 SymPy：

```powershell
python -m pip install -r requirements.txt
python main.py
```

也可以安装为命令行入口：

```powershell
python -m pip install -e .
hamivisualizer
```

注意：当前 GUI 运行时是 PySide6，不是 PyQt5；依赖明确要求 `numpy>=1.26,<2`，以避免 NumPy 2.x 与已测试 Qt 原生模块的 ABI 不兼容。

项目内的测试、诊断截图和临时数据统一放在 `.codex-artifacts/`，修改记录见 `docs/change-logs/`。

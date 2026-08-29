# HamiVisualizer 专属修改日志

此目录保存 Codex 持续打磨期间的逐项修改记录。每个功能或缺陷使用独立
Markdown 文件，记录真实日期时间、问题证据、实现思路、影响范围、验证命令、
截图和遗留风险。

约定：

- 文件名：`YYYY-MM-DD_HHMM_主题.md`；
- 时间均为 Asia/Shanghai（UTC+08:00）；
- 截图统一保存在项目内 `.codex-artifacts/screenshots/`；
- 历史诊断截图统一保存在项目内 `.codex-artifacts/diagnostics/`，诊断脚本统一保存在 `tools/diagnostics/`；
- 临时测试数据统一放在 `.codex-artifacts/test-data/`，不得读写真实
  `~/.hvisual` 用户工作区；
- 测试运行目录统一放在 `.codex-artifacts/test-runs/`，项目根目录不再新增测试临时目录；
- 只有测试与视觉证据均完成的功能，才能写入最终简报的“已验证功能”。

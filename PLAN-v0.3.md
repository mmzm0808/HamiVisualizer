# HamiVisualizer 路线图（0.3 之后）

旧 PLAN 混合了历史诊断与已完成功能；本文只列尚未完成的产品能力。

## Phase 1：内部自由度

- 轨道/自旋标签、矩阵跃迁、独立 `OnsiteTerm`；
- 基矢标签贯穿矩阵、晶格和波函数；
- [x] SSH 最小验证预设（交替胞内/胞间跃迁、OBC 端点态）；
- [x] Haldane 最小验证预设（蜂窝复次近邻相位、子格质量、能带开隙）；
- [ ] Haldane、Kane–Mele、BHZ、Kitaev 验证预设。

验收：解析能带或可信参考值对拍，旧标量 JSON/API 兼容。

## Phase 2：边界与拓扑

- 二维 PBC、圆柱、半元胞；
- Berry 曲率、Chern 数、Wilson loop、IPR、LDOS；
- 边缘态按局域化程度着色。

## Phase 3：科研工作流

- 图形化键编辑、方向相位、undo/redo、参数扫描；
- 波函数相位×幅度、能量筛选、colorbar；
- SVG/PDF 与论文级高分辨率输出。

## Phase 4：大规模后端

- SciPy 稀疏建阵和目标能窗求解；
- 可取消后台任务、进度报告和性能基准；
- 独立的非 Hermitian 求解路径。

验收必须给出参考硬件、时间、内存与交互延迟，不以主观“不卡”代替指标。

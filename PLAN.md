# HamiVisualizer v0.2 历史评审与路线图

> 当前路线图请阅读 [PLAN-v0.3.md](PLAN-v0.3.md)。以下内容是历史评审记录，不能作为当前状态清单。

> 状态：本文基于对 `hamivisualizer/` 全部源码、51 条测试（实测 **51 passed**）、
> MATLAB 参考 `HamiltonVisualizer.m`（1296 行）的完整通读写成。
> 结论先行：**架构方向是对的，数值也是对的；但「通用化」目前只做到「通用晶格」，
> 尚未做到「通用哈密顿量」。** 下一阶段的核心是引入**内部自由度（自旋/轨道/子格
> 作为可计算自由度）**，以及把缺掉的参数体系与拓扑诊断补上。

---

## 一、现状评估（优点，值得保留）

| # | 决策 | 为什么对 |
|---|------|---------|
| 1 | **数据驱动建模**：NP/SC 从独立建阵函数降级为 `presets.py` 里的一段数据 `(Lattice, [HoppingTerm])` | 新模型 = 写数据而非写代码，这是「自定义」的根基 |
| 2 | **DTO 视图解耦**：`rendermodel.py` 的 frozen dataclass，视图零模型依赖 | 视图可独立单测，泛化不再被 GUI 拖累 |
| 3 | **MATLAB 直译参考实现对拍**：`tests/reference_matlab.py` + `test_np_sc_matrix.py` 逐元 `1e-10` 对拍 | 数值正确性被钉死，后续重构有安全网 |
| 4 | **`fold_x` 地板语义 + `cs` 三去重规则**：负数安全，`H0/H1` 分类严谨 | 半无限建阵最容易错的地方，处理干净 |
| 5 | **符号模式 `real=True` + 自定义 `LatexPrinter`**：从表达式树保证 `t` 在 `e` 指数左边、无 `conj` | 比 MATLAB 的字符串 `strrep` 更结构化 |

这五点是对的地基，后续所有工作都应在它们之上叠加，不要推倒重来。

---

## 二、问题清单（按严重度排序）

### P0 — 功能回归 / 失效（先修，影响"能用"）

1. **符号模式在 GUI 路径下失效（已实测确认）**
   `load_preset()` 用 `factory(phi, 1.0, 1.0)` 生成模型后，把**数值**（`"-1"`、
   `"0.785398"`、`"1"`）写进表格（见 `controller.py` 的 `_amp_str/_phase_str`）。
   于是用户勾选「符号模式」后，`_build_hops()` 从表格读到的仍是数字，`sym_pretty`
   输出 `-1` 而非 `t e^{iφ}`。`HoppingTerm.name`（`"t"/"omg"`）是**死字段**，
   `evaluate()` 从不使用它。符号模式的单元测试之所以通过，是因为测试直接以
   `sympy.Symbol` 调用 `NP()`，绕开了 GUI 表格这一层。
   → 修复方向：presets 序列化时保留符号字符串（`"t"/"phi"/"omg"`），或让
   `name` 字段真正成为符号身份来源，并补一条「GUI 路径符号重建」的回归测试。

2. **丢掉了 MATLAB 的 `t/φ/ω` 参数控件（相对 MATLAB 的明显回退）**
   面板只有 `NX/NY/kx`。`t/phi/omg` 被硬编码在 `controller.DEFAULT_PARAMS`，
   **GUI 里无法改数值参数**——只能手动把某个跃迁的 amplitude 改成一个具体数字。
   MATLAB 版有 t/φ/ω 三组滑块。→ 引入可编辑的全局 `ParameterSet`（见 §四.1）。

3. **半无限能带与 `kx` 滑块脱节**
   `_push_band()` 总是画 `-π..π` 全谱，`kx` 只影响矩阵页标题/数值，能带图上
   **没有当前 `kx` 的竖线标记**；且每次拖 `kx` 都全量重算能带（浪费）。
   → 能带上叠加当前 `kx` 标记线；能带与矩阵解耦缓存。

4. **每次重建都 `fitInView` 重置视口**
   `_fit()` 在矩阵/晶格/能带三处每次 rebuild 都调用，用户缩放/平移后瞬间被重置，
   查看局部矩阵或晶格细节时不可用。→ 仅在首次或结构尺寸变化时 fit，否则保留
   当前变换。

### P1 — 物理通用性（核心扩展，决定"能不能叫哈密顿量工具"）

5. **无内部自由度（最重要的一条）**
   `Site` 只有 `(x, y, sublattice)`，`sublattice` 还是**仅显示配色**、不参与计算。
   `HoppingTerm` 是**标量**幅度。这决定了当前工具只能表达「单轨道标量跃迁」，
   而凝聚态/拓扑里几乎每个模型都需要自旋、轨道、子格作为**可计算**自由度：
   Haldane（2 分量）、SSH、Kane-Mele / BHZ（4×4 矩阵）、Rashba、p 波超导
   （BdG 粒子-空穴）……→ 见 §四.2，这是把「自定义晶格」升格为「自定义哈密顿量」
   的关键一跃。

6. **onsite 被 hack 成 `HoppingTerm(r, r)`**
   `presets.py` 用 `HoppingTerm("omg", r, r, (0,0), omg)` 表达对角项。它无法表达
   **子格依赖**的 onsite（交错质量 `m·σz`、Semenoff 质量），而这是拓扑相变的标配。
   → onsite 应成为独立的一等公民（`OnsiteTerm` / `Lattice.onsite[r]`），支持
   每格点矩阵。

7. **相位方向依赖（`directional`）未实现**
   `HoppingTerm.evaluate()` 只分支 `none` vs 其余，`"directional"` 会被**静默当
   `"phase"`** 处理；`applies_to` 是死字段。README 里承诺的「点键设相位」编辑器
   完全没做。

8. **仅支持 Hermitian**
   `eig()/wavefunctions()` 都用 `np.linalg.eigh`。非厄米（增益/损耗、趋肤效应、
   NH 拓扑）是「任意哈密顿量」的一大类，当前无法表达。

9. **边界只有 SEMI/OBC**
   README 承诺的 PBC（2D Bloch）、圆柱、半元胞（`halfCell`）都是桩：
   `RibbonSpec.edge_extra` 恒 0、无 UI 入口。

### P2 — UX / 功能缺口

10. 矩阵视图**没有行/列格点标签**（README §6.3 承诺，`_push_matrix` 未填 `sites`）。
11. 晶格视图**没有相位标注**（`t·e^{iφ}` 应画在键上，`lattice_view.py` 只画线）。
12. 点击矩阵元只写 statusBar（`_on_cell_clicked` 是 stub），无详情弹窗。
13. 波函数只有 `|ψ|²`：无相位可视化、无能量区间筛选、无 colorbar、无点格点探测。
14. 无保存/加载（JSON/YAML）、无撤销/重做；错误只进 statusBar，表格坏行被
    静默跳过（用户无感知）。
15. 能带无边缘态高亮 / 体隙标注 / 子带配色。

### P3 — 代码质量 / 性能

16. `controller.rebuild()` 第 148 行 `builder.build() if not symbolic else builder.build()`
    是**空三元**（死代码）。
17. `_sym_to_num()` 逐元素 `subs`，O(N²) 次 sympy 调用，符号模式大矩阵卡顿。
18. `build_obc/build_ribbon` 纯 Python 双循环 + dict 累加，无 `scipy.sparse`；
    能带每次全量重算，无缓存/后台线程，大系统 UI 会冻结。
19. `_build_lattice_scene` 是模块级函数却无直接测试；边类型用曼哈顿距离 ≤1 判
    NN/NNN，对任意晶格只是启发式，且无法区分「实键 vs 复相位键」。
20. 测试缺口：GUI 真实启动、save/load、能带/波函数视图渲染、`cs_far` 错误路径、
    负浮点 `fold_x`。

---

## 三、路线图（分阶段，每阶段含验收标准）

### Phase 0 — 修回归，先恢复"能用"（约 1–2 天）✅ 已完成 (v0.2)

- [x] 引入全局 `ParameterSet`（`t/φ/ω/...` 可编辑 + 滑块），替换 `DEFAULT_PARAMS`。
      —— 参数面板**自动收集**跃迁表达式中的自由符号生成控件，φ 按 φ/π 显示（MATLAB 同款）。
- [x] 修符号模式：预设以符号字符串（"-t"/"phi"/"omg"）写入表格，符号/数值共用同一份表格；
      任意自定义符号自动 real=True；补 GUI 路径回归测试。
- [x] `kx` 与能带联动：能带叠加当前 `kx` 红色标记线；签名缓存——只动 kx 走快路径（矩阵数值+标记线），能带/晶格不重算。
- [x] 保留视口变换（场景矩形+视口尺寸未变不 fit）；隐藏标签页 (尺寸 0) 跳过 fit，切页/显示时自动补 fit。
- [x] 清理死代码（空三元）；错误提升为面板红字标签；`directional` 显式报错。
- [x] **顺带修复（渲染正确性）**：
      - QPen 宽度默认是场景单位 → 全部改 cosmetic（恒像素宽，MATLAB LineWidth 语义）
      - 半透明 QBrush 在本环境渲染成黑 → 预混合到白底
      - OBC 晶格连线原来只画 from_site=0 的键 → 用 rmap 修正
      - 符号模式下能带/波函数崩溃（sympy 矩阵喂 numpy 对角化）→ lambdify 转数值
      - 虚影键 y 坐标错（起点误用终点 y）与 onsite 自环产生的零长度虚影键
- [x] **高保真晶格**：首胞黑粗框高亮、虚影层数按最大跃迁跨度自动（NP=1 层/SC=2 层，与 MATLAB 一致）、
      虚影键黯淡配色 (cNN*0.5+0.5 / cNNN*0.4+0.6)、矩阵行/列轴标号、点击矩阵元详情弹窗。
- [x] **保存/加载模型 JSON**（格点/跃迁/参数/边界/显示全量）+ 导出当前视图 PNG。
- **DoD**：✅ 62 条测试全绿（含 MATLAB 数值对拍 + 新回归）；
      ✅ 离屏像素级校验通过（64 格五类配色、A/B 子格、虚影、首胞边框、符号上标、SC 两层虚影、kx 标记）。

### Phase 1 — 内部自由度（架构升级，核心里程碑）

- [ ] `Site` 增加内部自由度（`orbitals: [labels]` 或维度 M）；基维度 = 格点数 × 轨道数。
- [ ] `HoppingTerm.amplitude/phase` 允许**矩阵**；`evaluate()` 返回矩阵，键贡献 =
      矩阵 + 共轭转置；onsite 独立成 `OnsiteTerm`（每格点矩阵，支持子格交错质量）。
- [ ] `HResult` 增加 `labels`（每基矢量的 `(格点, 轨道)`），矩阵/晶格/波函数统一消费
      ——顺带修复 P2-10 的矩阵无标签问题。
- [x] 新增 SSH 验证性 preset：交替胞内/胞间跃迁、OBC 近零能端点态，并完成数值回归；
- [x] 新增 Haldane 验证性 preset：蜂窝复次近邻相位、子格质量项和能带开隙回归；
- [ ] 新增 Haldane、Kane-Mele（4×4）、BHZ（4×4）、graphene、Kitaev，逐一与解析能带 /
      文献值对拍。
- [ ] 波函数视图支持自旋/轨道分辨（多通道 |ψ|² 或泡利投影）。
- **DoD**：Haldane 模型 2 分量能带 + 手征边缘态正确；BHZ 4×4 体隙 + 螺旋边缘态
  正确；所有新模型有对拍测试。

### Phase 2 — 边界与拓扑诊断

- [ ] PBC（2D Bloch `kx,ky` + `E(kx,ky)` 曲面/切片）、圆柱、半元胞（真正实现
      `edge_extra`）。
- [ ] 拓扑量：Berry 曲率 + Chern 数（Fukui–Hatsugai 离散法）、Wilson loop / 极化、
      体隙探测、局域化程度（IPR / 权重中心）、local DOS。
- [ ] 边缘态自动高亮（按局域化程度着色）。
- **DoD**：Haldane C=±1、Kane-Mele Z₂、SSH 极化能被工具自动算出并可视化。

### Phase 3 — UX 与可视化

- [ ] 晶格上点击键设置相位（`directional` 编辑器）+ 相位标注渲染（修 P2-11）。
- [ ] 矩阵行/列标签、点击弹窗详情（修 P2-12）。
- [ ] 波函数相位（hue）×幅度（亮度）极坐标着色、能量筛选、colorbar、点格点探测（修 P2-13）。
- [ ] 保存/加载 JSON、导出 PNG/SVG、undo/redo、参数扫描/动画。
- **DoD**：一个全新模型能在 1 分钟内用 UI 建好、看到带相位标注的晶格与能带、
  导出图片与可复现的 JSON。

### Phase 4 — 性能与健壮性

- [ ] `scipy.sparse` 建阵（OBC/ribbon 稀疏累加，对角化按需 `toarray` 或 `eigsh`）。
- [ ] `sp.lambdify` 向量化符号求值（修 P3-17）。
- [ ] 能带/波函数移到 `QThread` 后台 + 缓存失效，UI 不冻结。
- [ ] 非厄米支持（`eig` + 复能带 + 趋肤效应可视化，可选开关）。
- **DoD**：OBC 10×10（NP 400 格点）拖动参数实时刷新不卡；符号模式大矩阵秒级。

### Phase 5 — 模型库与论文级输出

- [ ] 内置模型库（Haldane / Kane-Mele / BHZ / SSH / Kitaev / graphene / 基础 TBG）。
- [ ] 一键导出 LaTeX 哈密顿量表达式（符号 TeX）与高分辨率图。
- [ ] （可选）与 LitTree 联动：从文献 KB 提取模型参数直接灌入。

---

## 四、建议的关键设计改动（具体到代码）

1. **`ParameterSet` 取代 `DEFAULT_PARAMS`**
   `class ParameterSet(Mapping[str, float|sympy.Symbol])`，是 controller 的单一事实
   来源：数值模式存 float，符号模式存 `param(name)`，面板可编辑，`_eval_expr` 与
   `ElementFormatter` 共用同一实例。修复 P0-2。

2. **内部自由度 + 矩阵跃迁**
   - `Site(r, x, y, sublattice, orbitals)`，`orbitals` 为该格点的内部轨道标签
     （如 `["A↑","A↓"]` 或维度 2）。
   - `HoppingTerm` 增加 `orbital_matrix`（或让 `amplitude` 接受 `np.ndarray|sympy.Matrix`）；
     `evaluate()` 返回矩阵，键贡献遵循 `H[j,i] = conj(H[i,j].T)`。
   - `OnsiteTerm(site, matrix)` 独立于 hopping；支持 `σz`/`σ0` 交错势。
   - `Lattice` 基维度 = `Σ orbitals(r)`；`Indexer` 的 `rmap` 携带轨道信息。

3. **`HResult.labels` 打通三视图**
   每个基矢量生成 `(格点, 轨道)` 标签，矩阵视图画轴标签、晶格视图画轨道、波函数
   视图分轨道着色——一处生成，处处消费，顺带修复矩阵无标签。

4. **`_build_lattice_scene` → `build_lattice_scene()` 提升为可测函数**
   搬到独立模块（如 `view/scene_builder.py`），补单测；边类型判定从「曼哈顿距离」
   改为「复用 `HoppingTerm` 元信息（phase_mode、cell_offset）」，能区分实键/复键。

5. **建阵层增加 `MatrixBuilder` 抽象**
   `数值稀疏 / 符号稠密 / 自动降级` 三种后端，`HResult` 只暴露统一接口；能带与
   波函数走后台线程 + 缓存失效。

---

## 五、下一步（Phase 0 已完成，见上方勾选清单）

按优先级：

1. **Phase 1 — 内部自由度**（核心里程碑）：`Site` 增加轨道/自旋维度、跃迁幅度支持矩阵、
   onsite 独立为每格点矩阵（子格交错质量）、`HResult.labels` 打通三视图；新增 Haldane/SSH/
   Kane-Mele/BHZ/graphene/Kitaev 预设并对拍。
2. **Phase 2 — 拓扑诊断**：PBC（kx,ky）、圆柱、半元胞；Berry 曲率 + Chern 数、Wilson loop、
   边缘态局域化高亮。
3. **Phase 3 — UX**：晶格上点键设相位（`directional` 编辑器）、相位标注渲染、波函数相位×幅度着色、
   colorbar、undo/redo。
4. **Phase 4 — 性能**：`scipy.sparse` 建阵、能带/波函数后台线程、非厄米支持。

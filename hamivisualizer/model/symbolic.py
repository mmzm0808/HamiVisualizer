"""符号模式: SymPy 重建 H 与可读 TeX 显示.

翻译 MATLAB sym_pretty 的全部规则 (README §2.3 / 附录):
  - exp(±…) → e^{±…};  kx → k_x 下标
  - t / tc / ω 永远放在所有 e 指数最左 (因子分组, 结构保证)
  - 实参数 (Symbol(..., real=True)) 自动化简, 无 conj
  - 长表达式在 e 指数间换行并保留减号 (wrap_tex)
"""

from __future__ import annotations

import re

import sympy as sp
from sympy.printing.latex import LatexPrinter

from .expression import collect_symbols

# 参数符号名 → TeX
SPECIAL_TEX = {
    "kx": "k_{x}",
    "k_x": "k_{x}",
    "omg": r"\omega",
    "omega": r"\omega",
    "phi": r"\phi",
    "tc": "t_{c}",
}


def param(name: str):
    """创建实符号参数 (real=True 从根源消除 conj)."""
    return sp.Symbol(name, real=True)


def combine_semi(H0, H1, kx):
    """符号 H(kx) = H0 + H1·e^{ikx} + H1†·e^{-ikx}.

    与 MATLAB H0 + H1*exp(1i*kx) + H1'*exp(-1i*kx) 逐元一致。
    """
    ph = sp.exp(sp.I * kx)
    return H0 + ph * H1 + H1.T.conjugate() * sp.exp(-sp.I * kx)


def _is_single_symbol(s: str) -> bool:
    """渲染串是否代表单个符号 (无需括号)."""
    return bool(re.fullmatch(r"[A-Za-z_\\][A-Za-z0-9_{}]*", s))


class PrettyLatexPrinter(LatexPrinter):
    """自定义 Latex 打印器.

    _print_Mul: 因子分组 —— 数值系数(I 合并) + 非 e 符号在前, e 指数全部最后,
                从表达式树保证 t/tc/ω 永远在 e 指数最左。
    _print_exp: 提取纯虚系数与实线性组合 → e^{±i·(...)} 任意线性指数。
    _print_Symbol: kx → k_x, omg → ω (TeX) 等。
    """

    def _print_Symbol(self, expr):
        special = SPECIAL_TEX.get(expr.name)
        if special is not None:
            return special
        # Parameter tables conventionally use compact names such as t1/t2,
        # lambda2 or g12.  Matrix mathematics should typeset the numerical
        # suffix as a true subscript instead of an ordinary baseline digit.
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", expr.name)
        if match:
            return f"{match.group(1)}_{{{match.group(2)}}}"
        return expr.name

    def _print_exp(self, expr):
        arg = sp.expand(expr.exp)
        if arg == 0:
            return "1"
        if arg.has(sp.I):
            c = sp.expand(sp.re(sp.expand(arg / sp.I)))
            body = self._print(c)
            if body.startswith("-"):
                inner = body[1:].strip()
            else:
                inner = body
            if not _is_single_symbol(inner):
                inner = r"\left(" + inner + r"\right)"
            if body.startswith("-"):
                return r"e^{-i" + inner + "}"
            return r"e^{i" + inner + "}"
        return r"e^{" + self._print(arg) + "}"

    def _print_conjugate(self, expr):
        # 只对可证明为实数的表达式消除共轭；未知复性必须忠实保留。
        if expr.args[0].is_real is True:
            return self._print(expr.args[0])
        return super()._print_conjugate(expr)

    def _print_Mul(self, expr):
        coeff = expr.as_coeff_Mul()[0]
        exps = []
        others = []
        has_I = False
        for f in sp.Mul.make_args(expr):
            if f.is_Number:
                continue  # 数值已并入 coeff
            if f is sp.I:
                has_I = True
            elif isinstance(f, sp.exp):
                exps.append(f)
            else:
                others.append(f)
        s = self._print_coeff_and_i(coeff, has_I, exps or others)
        for f in others:
            s += self._print(f) + " "
        for f in exps:
            s += self._print(f) + " "
        return s.rstrip()

    def _print_coeff_and_i(self, coeff, has_I, has_factors):
        if coeff == 1 and not has_I:
            return ""
        if coeff == -1 and not has_I:
            return "- "
        sign = "- " if (hasattr(coeff, "is_negative") and coeff.is_negative) else ""
        mag = -coeff if sign else coeff
        c = "" if mag == 1 else self._print(sp.nsimplify(mag))
        body = c + ("i" if has_I else "")
        if not body:
            return sign
        return sign + body + " "

    def _print_ImaginaryUnit(self, expr):
        return "i"


def sym_pretty(expr) -> str:
    """符号表达式 → 可读 TeX 字符串."""
    printer = PrettyLatexPrinter(settings={})
    return printer.doprint(expr)


class ElementFormatter:
    """矩阵元智能识别 (翻译 MATLAB format_elem, §2.2).

    把数值矩阵元识别为物理参数表达式串 (Unicode):
      对角: n·ω / ω±n·t;  实: ±n·t;  复: ±n·t·e^{±iφ};
      trig: ±2t·cosφ / ±2it·sinφ;  兜底数值。
    优先级与 MATLAB 一致: 对角 > ±n·t > ±n·t·e^{±iφ} > trig。
    """

    def __init__(self, t=1.0, phi=None, omg=None, max_mult: int = 8, tol: float = 1e-8):
        import math

        self.t = t
        self.phi = phi
        self.omg = omg
        self.max_mult = max_mult
        self.tol = tol
        self._math = math

    def format(self, value, is_diag: bool = False) -> str:
        import cmath

        v = complex(value)
        if abs(v) < self.tol:
            return "0"
        t, omg = self.t, self.omg
        # 对角: n·ω
        if is_diag and omg is not None:
            for n in range(self.max_mult, 0, -1):
                if abs(v - n * omg) < self.tol:
                    return f"{n}ω" if n > 1 else "ω"
            for n in range(self.max_mult, 0, -1):
                if abs(v - (omg + n * t)) < self.tol:
                    return f"ω+{n}t"
                if abs(v - (omg - n * t)) < self.tol:
                    return f"ω-{n}t" if n > 1 else "ω-t"
        # ±n·t
        for n in range(self.max_mult, 0, -1):
            if abs(v - n * t) < self.tol:
                return f"{n}t" if n > 1 else "t"
            if abs(v + n * t) < self.tol:
                return f"-{n}t" if n > 1 else "-t"
        # ±n·t·e^{±iφ}
        if self.phi is not None:
            e = cmath.exp
            bases = {
                t * e(1j * self.phi): "t·e^{iφ}",
                t * e(-1j * self.phi): "t·e^{-iφ}",
            }
            for n in range(self.max_mult, 0, -1):
                for b, nm in bases.items():
                    if abs(v - n * b) < self.tol:
                        return f"{n}·{nm}" if n > 1 else nm
                    if abs(v + n * b) < self.tol:
                        return f"-{n}·{nm}" if n > 1 else f"-{nm}"
            # trig 组合
            c = 2 * t * self._math.cos(self.phi)
            s = 2 * t * self._math.sin(self.phi)
            if abs(v.imag) < self.tol and abs(v.real - c) < self.tol:
                return "2t·cosφ"
            if abs(v.real) < self.tol and abs(v.imag - s) < self.tol:
                return "2it·sinφ"
            if abs(v.imag) < self.tol and abs(v.real + c) < self.tol:
                return "-2t·cosφ"
            if abs(v.real) < self.tol and abs(v.imag + s) < self.tol:
                return "-2it·sinφ"
        # 兜底数值
        if abs(v.imag) < self.tol:
            return f"{v.real:.2f}"
        return f"{v.real:.2f}{v.imag:+.2f}i"


def format_elem(value, mode: str = "smart", is_diag: bool = False,
                formatter: ElementFormatter | None = None) -> str:
    """矩阵元 → 显示串.

    mode: 'smart' 智能识别 | 'numeric' 直接 %.2f | 'symbolic' 用符号串。
    符号模式下 value 为 sympy 表达式, 用 sym_pretty。
    """
    if mode == "symbolic":
        return sym_pretty(value) if value != 0 else "0"
    if mode == "numeric":
        v = complex(value)
        return f"{v.real:.2f}" if abs(v.imag) < 1e-12 else f"{v.real:.2f}{v.imag:+.2f}i"
    fm = formatter or ElementFormatter()
    return fm.format(value, is_diag)


def format_bloch_elem(expr, formatter: ElementFormatter | None = None,
                      is_diag: bool = False) -> str:
    """Compactly format a numeric-coefficient Bloch expression.

    ``expr`` may contain one or more ``exp(i*n*kx)`` harmonics while all model
    parameters have already been evaluated numerically.  Coefficients are fed
    through :class:`ElementFormatter` first, so the UI shows ``t``/``phi``
    semantics instead of binary-float artefacts such as
    ``0.707106781186548 - sqrt(2)/2*i``.
    """
    fm = formatter or ElementFormatter()
    if not isinstance(expr, sp.Basic):
        return fm.format(expr, is_diag)
    kx_symbols = [s for s in expr.free_symbols if s.name in {"kx", "k_x"}]
    if not kx_symbols:
        try:
            return fm.format(complex(expr.evalf()), is_diag)
        except (TypeError, ValueError):
            return sym_pretty(expr)
    kx = kx_symbols[0]
    grouped: dict[sp.Expr, sp.Expr] = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        exponentials = [e for e in term.atoms(sp.exp) if e.has(kx)]
        if len(exponentials) > 1:
            return sym_pretty(expr)
        phase = exponentials[0] if exponentials else None
        coefficient = sp.simplify(term / phase) if phase is not None else term
        harmonic = (sp.simplify(phase.args[0] / (sp.I * kx))
                    if phase is not None else sp.Integer(0))
        grouped[harmonic] = sp.simplify(grouped.get(harmonic, 0) + coefficient)
    parts: list[str] = []
    for harmonic, coefficient in grouped.items():
        try:
            label = fm.format(complex(coefficient.evalf()), is_diag and harmonic == 0)
        except (TypeError, ValueError):
            return sym_pretty(expr)
        if label == "0":
            continue
        if harmonic != 0:
            if harmonic.is_integer:
                n = int(harmonic)
                if n == 1:
                    phase_label = "e^{ik_{x}}"
                elif n == -1:
                    phase_label = "e^{-ik_{x}}"
                else:
                    phase_label = f"e^{{{n}ik_{{x}}}}"
            else:
                phase_label = sym_pretty(phase)
            label = f"{label}·{phase_label}"
        parts.append(label)
    if not parts:
        return "0"
    result = parts[0]
    for part in parts[1:]:
        result += f" - {part[1:]}" if part.startswith("-") else f" + {part}"
    return result


def wrap_tex(s: str, max_len: int = 14) -> str:
    """长表达式在 e 指数间换行并保留减号 (MATLAB §2.3).

    当字符串含 ≥2 个 e^{...} 且长度超阈值时, 在第二个 e 前断行,
    若行内出现 ' - ' 则从减号处断, 减号保留行首。
    """
    if len(s) <= max_len:
        return s
    eidx = [m.start() for m in re.finditer(r"e\^", s)]
    if len(eidx) < 2:
        return s
    brk = eidx[1]
    pm = [m.start() for m in re.finditer(r" - ", s[:brk])]
    if pm:
        brk = pm[-1]
        return s[: brk + 1] + "\n" + s[brk + 1 :]
    return s[:brk] + "\n" + s[brk:]


def to_html(s: str) -> str:
    """TeX 串 → Qt 富文本 (HTML 子集): 模仿 MATLAB tex 解释器的观感.

      e^{...} → e<sup>...</sup>      (支持一层嵌套花括号)
      _{...}  → <sub>...</sub>      (如 k_{x})
      \\left( \\right) → 普通括号
      \\omega → ω, \\phi → φ
    """
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    # Keep ``x`` as a true subscript of k *inside* the exponential.  The
    # previous conversion moved k outside the exponent for braced input and
    # missed the common ``k_x`` spelling, making x look like an exponent-level
    # character instead of a subscript.
    s = re.sub(
        r"e\^\{(-?)(\d*)i(\s*)k_(?:\{x\}|x)\}",
        lambda m: f"e<sup>{m.group(1)}{m.group(2)}i{m.group(3)}k<sub>x</sub></sup>",
        s,
    )
    s = re.sub(r"e\^\{((?:[^{}]|\{[^{}]*\})*)\}", r"e<sup>\1</sup>", s)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = s.replace(r"\omega", "ω").replace(r"\phi", "φ")
    return s


def to_math_html(s: str) -> str:
    """Render a TeX-like expression with a math-capable typeface.

    ``to_html`` remains the compatibility conversion used by integrations
    that compare its compact HTML fragment.  Matrix cells use this richer
    wrapper so the expression does not inherit the UI's sans-serif CJK font.
    Cambria Math is available on standard Windows installations; Qt falls
    back to the listed serif faces when it is not.
    """
    rendered = to_html(s)
    # Qt rich text does not implement TeX's nested-script layout.  With the
    # natural HTML translation ``<sup>ik<sub>x</sub></sup>`` it lowers ``x``
    # once relative to the superscript *and then again* relative to the base
    # line, which is why ``e^{ik_x}`` previously looked visibly broken.  The
    # mathematical Unicode subscript is part of the exponent run instead:
    # Cambria Math/STIX typesets it with the intended compact, raised optical
    # position, while the complete ``ikₓ`` group remains the exponent of e.
    # Keep ``to_html`` unchanged as its literal tags are a compatibility API.
    rendered = re.sub(
        r"<sup>((?:-?\d*)i\s*k)<sub>x</sub></sup>",
        r"<sup>\1ₓ</sup>",
        rendered,
    )
    # U+2212 has the optical width and vertical alignment expected in display
    # mathematics; keep ASCII '-' in ``to_html`` for API compatibility.
    rendered = rendered.replace("-", "−")
    return (
        '<span style="font-family: \'Cambria Math\', \'STIX Two Math\', '
        '\'DejaVu Serif\', serif;">' + rendered + '</span>'
    )


def collect_param_names(expr_strings) -> set[str]:
    """从表达式字符串表收集用户参数名 (自由符号), 排除常量.

    用于自动生成参数面板: 表格里出现的每个符号 (t/phi/omg/任意自定义名)
    都会得到一个可编辑的数值控件。
    """
    return collect_symbols(expr_strings)

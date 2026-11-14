# -*- coding: utf-8 -*-
"""建模工具库：把三个建模 Notebook 里的函数抽成一个可复用模块。

来源
----
- ``模型/层次分析法.ipynb``   → AHP（特征值法 / 算术平均法 / 几何平均法）
- ``模型/熵权法.ipynb``       → 指标正向化 + 熵权法 + TOPSIS
- ``模型/多元线性回归.ipynb`` → OLS 回归、VIF、预测

函数体均与 Notebook 中的版本保持一致，Notebook 仍保留完整讲解，
本模块只负责"让新题目不用再复制粘贴"。

用法
----
Notebook 与本文件放在同一目录时（例如都在 ``模型/`` 下）::

    import 建模工具
    w, CR = 建模工具.ahp(A, names=["景色", "费用", "居住", "饮食"])

放在其他目录时，先把 ``模型`` 加进搜索路径::

    import sys, pathlib
    sys.path.append(str(pathlib.Path.cwd().parent / "模型"))
    import 建模工具

依赖
----
numpy、pandas 必需；scipy 只有多元线性回归需要。
"""

__version__ = "1.0"

__all__ = [
    # AHP
    "RI",
    "ahp",
    "consistency_check",
    "ahp_arithmetic",
    "ahp_geometric",
    # 熵权法 + TOPSIS
    "KINDS",
    "positivize",
    "entropy_weight",
    "topsis",
    "ewm_topsis",
    # 多元线性回归
    "add_intercept",
    "calculate_vif",
    "fit_multiple_linear_regression",
    "predict_multiple_linear_regression",
]

import numpy as np
import pandas as pd

try:                                   # scipy 只有多元线性回归用得到，缺了不影响 AHP / 熵权法
    from scipy import stats
except ImportError:                    # pragma: no cover
    stats = None


# =====================================================================
# 一、层次分析法 AHP
# =====================================================================

# 随机一致性指标 RI（n = 1~10）
RI = [0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]


def ahp(matrix, names=None, verbose=True):
    """特征值法求权重 + 一致性检验，返回 (w, CR)。

    w 为归一化后的权重向量（和为 1），CR 为一致性比例（< 0.1 视为通过）。
    RI 表只覆盖 n ≤ 10，指标更多时需要自行补充 RI 取值。
    """
    A = np.asarray(matrix, dtype=float)
    n = A.shape[0]

    # 合法性检查（方阵、对角线为 1、互反性）
    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("判断矩阵必须是方阵")
    if not np.allclose(np.diag(A), 1):
        raise ValueError("对角线元素必须为 1")
    if not np.allclose(A * A.T, 1):
        raise ValueError("必须满足互反性：a_ij * a_ji = 1")
    #非方阵不可用AHP方法，且对角线必须为1，互反性是AHP的基本要求

    # 1. 最大特征值及其对应的特征向量
    eigvals, eigvecs = np.linalg.eig(A)#前面特征值，后面特征向量
    k = int(np.argmax(eigvals.real))#找到最大特征值索引
    lambda_max = eigvals[k].real#找到最大特征值实部
    w = np.abs(eigvecs[:, k].real)#获取最大特征值对应的特征向量绝对值
    w = w / w.sum()# 归一化得到权重，保证权重和为1
    # 2. 一致性检验
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    CR = CI / RI[n - 1] if RI[n - 1] > 0 else 0.0#else做保护机制

    # 3. 输出结果
    if verbose:
        names = names or [f"准则{i + 1}" for i in range(n)]
        for name, wi in zip(names, w):
            print(f"{name:<6s} 权重 = {wi:.4f}  {'█' * int(round(wi * 40))}")
        print(f"λmax = {lambda_max:.4f}, CI = {CI:.4f}, RI = {RI[n - 1]:.2f}, CR = {CR:.4f}")
        print("一致性检验：" + ("通过 ✅" if CR < 0.1 else "未通过 ❌，请调整判断矩阵"))

    return w, CR


def consistency_check(A, w):
    """由权重 w 反推 lambda_max，返回 (lambda_max, CI, CR)。

    特征值法可直接得到 λmax，而算术平均法 / 几何平均法只给出权重 w，
    所以用 λmax ≈ (1/n) * Σ (A*w)_i / w_i 反推（矩阵越一致，该近似越准）。
    """
    A = np.asarray(A, dtype=float)       # 统一转成浮点数组，避免整数相除被截断
    n = A.shape[0]                       # 矩阵阶数
    lambda_max = np.mean((A @ w) / w)    # A @ w 是 A 乘权重向量；逐元素除以 w 后取平均
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0   # 完全一致时 λmax = n ⇒ CI = 0
    CR = CI / RI[n - 1] if RI[n - 1] > 0 else 0.0       # RI 为 0（1、2 阶）时直接记 0
    return lambda_max, CI, CR


def _summary(w, A, names, verbose):
    """统一打印权重条形图和一致性检验结果，返回 (w, CR)，给下面两个方法共用。"""
    lambda_max, CI, CR = consistency_check(A, w)
    if verbose:
        # 没给名字就自动生成“准则1…准则n”，保证与 w 长度一致
        names = names or [f"准则{i + 1}" for i in range(len(w))]
        for name, wi in zip(names, w):
            # 方块数量粗略表示权重大小（满格 = 权重 1.0，即 40 个方块）
            print(f"{name:<6s} 权重 = {wi:.4f}  {'█' * int(round(wi * 40))}")
        print(f"λmax = {lambda_max:.4f}, CI = {CI:.4f}, RI = {RI[len(w) - 1]:.2f}, CR = {CR:.4f}")
        print("一致性检验：" + ("通过 ✅" if CR < 0.1 else "未通过 ❌，请调整判断矩阵"))
    return w, CR


def ahp_arithmetic(matrix, names=None, verbose=True):
    """算术平均法求权重：每列先归一化，再按行取平均。

    公式：w_i = (1/n) * Σ_j a_ij / Σ_k a_kj
    直觉：对完全一致的判断矩阵，任意一列归一化后都正好等于权重向量，
          因此对 n 列取平均可以互相抵消个别的打分误差。
    """
    A = np.asarray(matrix, dtype=float)   # 转成浮点数组
    col_sum = A.sum(axis=0)               # axis=0 表示按列求和，得到每一列的元素之和
    w = (A / col_sum).mean(axis=1)        # 每列除以列和（列归一化），再按行求平均
    return _summary(w, A, names, verbose)


def ahp_geometric(matrix, names=None, verbose=True):
    """几何平均法（方根法）求权重：每行求几何平均后归一化。

    公式：w_i = (∏_j a_ij)^(1/n) / Σ_k (∏_j a_kj)^(1/n)
    直觉：用乘积代替求和，更接近特征向量（Perron 向量是矩阵幂的极限），
          但对个别极端打分也更敏感。
    """
    A = np.asarray(matrix, dtype=float)   # 转成浮点数组
    n = A.shape[0]                        # 阶数，也是开方的次数
    w = np.prod(A, axis=1) ** (1 / n)     # axis=1 表示按行连乘，再开 n 次方（行几何平均）
    w = w / w.sum()                       # 归一化，使权重之和为 1
    return _summary(w, A, names, verbose)


# =====================================================================
# 二、熵权法 + TOPSIS
# =====================================================================

# 四种指标类型（正向化的四种公式）：
#   benefit      极大型（效益型）：数值越大越好，例如 GDP、教育投入
#   cost         极小型（成本型）：数值越小越好，例如能耗、污染排放
#   intermediate 中间型：越接近某个最优值越好，例如最适宜气温 22℃
#   interval     区间型：落在某个最优区间内最好，例如绿化覆盖率 40%~50%
KINDS = ("benefit", "cost", "intermediate", "interval")


def positivize(X, kinds, params=None):
    """把所有指标统一正向化为"越大越好"。

    X      : (n_samples, m_indicators) 原始数据（行=样本，列=指标）
    kinds  : 长度 m 的列表，元素取自 KINDS，说明每一列的类型
    params : 与 kinds 一一对应的参数
             intermediate -> x_best（最优值，一个数）
             interval     -> (a, b)（最优区间的左右端点）
             其他类型不需要参数，填 None 即可
    返回正向化后的矩阵（与 X 同形状，每列都满足"越大越好"）
    """
    # 统一转成 float 数组，避免整数除法、类型不一致等隐患
    X = np.asarray(X, dtype=float)

    # ---------- 先做一组参数校验：出错时给出明确提示，方便排查 ----------
    if X.ndim != 2:                       # 必须二维：行是样本，列是指标
        raise ValueError("X 必须是二维矩阵（样本 × 指标）")
    if len(kinds) != X.shape[1]:          # 每个指标都要说明自己的类型
        raise ValueError("kinds 长度必须等于指标个数")
    if not set(kinds) <= set(KINDS):      # 类型只能是上面那四种之一
        raise ValueError(f"指标类型只能是 {KINDS}")

    # params 允许不传，不传就当成"全为 None"
    params = list(params) if params is not None else [None] * len(kinds)
    if len(params) != len(kinds):
        raise ValueError("params 长度必须等于指标个数")

    # 结果矩阵：先复制原始数据，后面逐列覆盖成正向化后的值
    Y = X.copy()
    # 一次算出每一列的最大值（成本型公式 y = max(x) - x 需要用到）
    col_max = X.max(axis=0)

    # 逐列（逐个指标）处理：j 是列号，kind 是该列的类型
    for j, kind in enumerate(kinds):
        col = X[:, j]                     # 取出第 j 列（所有样本在该指标上的取值）

        if kind == "benefit":
            # 极大型：本来就越大越好，原样保留
            Y[:, j] = col

        elif kind == "cost":
            # 极小型：用「该列最大值 - 原值」把方向翻转，原来越小现在越大
            Y[:, j] = col_max[j] - col

        elif kind == "intermediate":
            # 中间型：离最优值 x_best 越近越好
            # gap = |x - x_best| 是偏差，再把偏差除以最大偏差压到 [0, 1]
            x_best = float(params[j])
            gap = np.abs(col - x_best)
            top = gap.max()
            # top == 0 表示所有样本都恰好等于最优值，此时统一给满分 1
            Y[:, j] = 1.0 - gap / top if top > 0 else 1.0

        else:  # interval：区间型，落在 [a, b] 之内最好
            a, b = params[j]
            lo, hi = min(a, b), max(a, b)   # 防止把区间端点写反
            below = np.where(col < lo, lo - col, 0.0)   # 低于下界的距离（区间内算 0）
            above = np.where(col > hi, col - hi, 0.0)   # 高于上界的距离（区间内算 0）
            dist = below + above                        # 到区间的最短距离
            top = dist.max()
            # 距离为 0 → 得分 1（正好在区间内）；距离最大 → 得分 0
            Y[:, j] = 1.0 - dist / top if top > 0 else 1.0

    # 返回的矩阵每一列都是"越大越好"，可以直接交给熵权法
    return Y


def entropy_weight(X):
    """熵权法求权重，要求 X 已正向化且非负。

    返回 (w, Z, e)：
      w —— 各指标权重，所有列权重之和为 1
      Z —— 向量归一化后的矩阵，供 TOPSIS 继续使用
      e —— 各指标信息熵（越接近 1 说明该指标在各样本间差异越小、权重越低）
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵")
    if (X < 0).any():
        # 熵权法里要算 p * ln(p)，要求数据非负，所以必须先正向化
        raise ValueError("熵权法要求数据非负，请先用 positivize 正向化")
    n, m = X.shape          # n = 样本个数（如城市数），m = 指标个数
    if n < 2:
        raise ValueError("样本数至少为 2")

    # 1. 向量归一化：每列除以该列的"向量长度"，消除不同量纲带来的影响
    #    np.linalg.norm(X, axis=0) 等价于对每一列求 sqrt(Σ x²)
    #    where 用来防止整列全为 0 时出现除以 0（这种情况直接除以 1）
    norm = np.linalg.norm(X, axis=0)
    Z = X / np.where(norm == 0, 1.0, norm)

    # 2. 计算比重 p_ij：第 j 个指标下，第 i 个样本占该指标总和的比例
    #    处理完之后，每一列加起来都等于 1
    col_sum = Z.sum(axis=0)
    P = Z / np.where(col_sum == 0, 1.0, col_sum)

    # 3. 信息熵 e_j = -(1/ln n) * Σ p_ij * ln(p_ij)
    #    约定 0 * ln 0 = 0，所以先用 where 把 p == 0 的位置替换成 0 再求和
    #    直觉：某个指标各样本取值越平均 → 熵越大 → 提供的信息越少 → 权重应越小
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(P > 0, P * np.log(P), 0.0)
    e = -L.sum(axis=0) / np.log(n)

    # 4. 差异系数 d = 1 - e：熵越小（各样本差异越大）→ d 越大 → 最终权重越大
    #    再归一化，让所有指标的权重之和等于 1
    d = 1.0 - e
    total = d.sum()
    # 极端情况：所有列取值完全相同 → 每个 e 都是 1 → d 全为 0，此时退化为等权
    w = d / total if total > 0 else np.full(m, 1.0 / m)
    return w, Z, e


def topsis(Z, w):
    """加权 TOPSIS，输入为标准化矩阵 Z 与权重 w。

    返回 (C, d_pos, d_neg, v_best, v_worst)：
      C        —— 贴近度，取值 [0, 1]，越大说明越接近理想方案
      d_pos    —— 到正理想解的距离
      d_neg    —— 到负理想解的距离
      v_best   —— 正理想解（每个指标的最优值）
      v_worst  —— 负理想解（每个指标的最差值）
    """
    Z = np.asarray(Z, dtype=float)
    w = np.asarray(w, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != w.size:
        raise ValueError("Z 的列数必须与权重个数一致")

    # 1. 加权：每个指标乘上自己的权重，重要指标对结果影响更大
    V = Z * w                                       # 加权规范化矩阵
    # 2. 构造理想解：每一列的最大值就是该指标的"最好水平"，最小值是"最差水平"
    v_best = V.max(axis=0)                          # 正理想解 v+
    v_worst = V.min(axis=0)                         # 负理想解 v-

    # 3. 每个样本到正/负理想解的欧氏距离（先按行求平方和，再开根号）
    d_pos = np.sqrt(((V - v_best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - v_worst) ** 2).sum(axis=1))

    # 4. 贴近度 C = d_neg / (d_pos + d_neg)：越靠近正理想解，C 越接近 1
    #    where 防止分母为 0 时抛出除零错误
    denom = d_pos + d_neg
    C = np.divide(d_neg, denom, out=np.zeros_like(d_neg), where=denom > 0)

    # 退化情形：正负理想解重合（所有方案完全相同）时，一律算满分 1
    C = np.where(denom > 0, C, 1.0)
    return C, d_pos, d_neg, v_best, v_worst


def ewm_topsis(X, kinds, params=None, names=None, index=None, show=True):
    """熵权法 + TOPSIS 一体化评价，返回结果 DataFrame（按贴近度降序）。

    内部流程一共三步：正向化 → 熵权法定权重 → TOPSIS 打分排名
    """
    X = np.asarray(X, dtype=float)
    m = X.shape[1]      # 指标个数
    # 没传指标名就用「指标1、指标2……」占位
    names = list(names) if names is not None else [f"指标{j + 1}" for j in range(m)]

    # 第 1 步：正向化，把所有指标统一成"越大越好"
    Y = positivize(X, kinds, params)
    # 第 2 步：熵权法，得到权重 w、标准化矩阵 Z、信息熵 e
    w, Z, e = entropy_weight(Y)
    # 第 3 步：TOPSIS，得到每个样本的贴近度 C 与到两个理想解的距离
    C, d_pos, d_neg, _, _ = topsis(Z, w)

    if show:
        # 把权重相关的中间结果打印出来，方便检查"某指标为什么权重高/低"
        table = pd.DataFrame({
            "指标": names,
            "类型": kinds,
            "信息熵 e": e,
            "差异系数 1-e": 1.0 - e,
            "权重 w": w,
        }).set_index("指标")
        print("=== 熵权法权重 ===\n")
        print(table.round(4).to_string())

    # 组装最终结果表：index 是样本名称（如城市名）
    result = pd.DataFrame({
        "贴近度 C": C,
        "正理想解距离": d_pos,
        "负理想解距离": d_neg,
    }, index=index)
    # rank(ascending=False) 让得分最高的排第 1；method="min" 表示并列时取较小名次
    result["排名"] = result["贴近度 C"].rank(ascending=False, method="min").astype(int)
    # 按贴近度从高到低排序，第一名排在最上面
    return result.sort_values("贴近度 C", ascending=False)


# =====================================================================
# 三、多元线性回归
# =====================================================================

def add_intercept(features):
    """在特征矩阵最左侧添加一列 1，用于估计截距项。

    参数
    ----
    features : 二维数组或 DataFrame，形状为 (n_samples, n_features)

    返回
    ----
    design_matrix : 形状为 (n_samples, n_features + 1) 的二维数组
    """
    feature_matrix = np.asarray(features, dtype=float)

    # 一维输入按“一个特征”处理；实际回归通常应传入二维矩阵。
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(-1, 1)
    if feature_matrix.ndim != 2:
        raise ValueError('features 必须是二维矩阵')

    # column_stack 会把全 1 列和原始特征按列拼接。
    ones = np.ones((feature_matrix.shape[0], 1), dtype=float)
    return np.column_stack((ones, feature_matrix))


def calculate_vif(features, feature_names):
    """计算每个特征的方差膨胀因子 VIF。

    VIF_j 等于把第 j 个特征作为因变量、其余特征作为自变量做回归后得到的
    1 / (1 - R_j^2)。VIF 越大，说明该特征越容易被其他特征线性解释。
    经验判断：VIF > 5 需关注，VIF > 10 通常认为共线性较严重。
    """
    feature_matrix = np.asarray(features, dtype=float)
    sample_count, feature_count = feature_matrix.shape
    vif_values = np.ones(feature_count, dtype=float)

    # 只有一个特征时不存在“由其他特征解释”的问题，VIF 记为 1。
    if feature_count == 1:
        return pd.DataFrame({'特征': feature_names, 'VIF': vif_values})

    for j in range(feature_count):
        target_j = feature_matrix[:, j]
        other_columns = np.delete(feature_matrix, j, axis=1)
        auxiliary_design = add_intercept(other_columns)

        # 辅助回归：用其他特征尽量解释当前特征。
        auxiliary_coefficients, *_ = np.linalg.lstsq(
            auxiliary_design, target_j, rcond=None
        )
        auxiliary_prediction = auxiliary_design @ auxiliary_coefficients
        auxiliary_residuals = target_j - auxiliary_prediction

        total_sum_squares = np.sum((target_j - target_j.mean()) ** 2)
        residual_sum_squares = np.sum(auxiliary_residuals ** 2)

        if total_sum_squares <= np.finfo(float).eps:
            # 特征本身是常数，无法进行有意义的共线性分析。
            vif_values[j] = np.inf
            continue

        auxiliary_r_squared = 1.0 - residual_sum_squares / total_sum_squares
        # 数值误差可能使 R² 略大于 1，先裁剪到合理区间。
        auxiliary_r_squared = np.clip(auxiliary_r_squared, 0.0, 1.0 - 1e-12)
        vif_values[j] = 1.0 / (1.0 - auxiliary_r_squared)

    return pd.DataFrame({'特征': feature_names, 'VIF': vif_values})


def fit_multiple_linear_regression(features, target, feature_names=None,
                                   confidence=0.95):
    """拟合多元线性回归，并返回系数、显著性检验和拟合优度。

    参数
    ----
    features : 二维数组或 DataFrame，形状为 (n_samples, n_features)
    target : 一维数组或 Series，形状为 (n_samples,)
    feature_names : 特征名称列表，默认自动命名为 x1、x2……
    confidence : 回归系数置信区间的置信水平

    返回
    ----
    字典，包含系数表、拟合值、残差、模型指标和置信水平。
    """
    if stats is None:      # 本模块把 scipy 的导入做成了可选
        raise ImportError('多元线性回归需要 scipy，请先运行 pip install scipy')

    feature_matrix = np.asarray(features, dtype=float)
    target_vector = np.asarray(target, dtype=float).reshape(-1)

    # 输入检查能在数据错误时给出明确提示，避免矩阵运算抛出难懂的错误。
    if feature_matrix.ndim != 2:
        raise ValueError('features 必须是二维矩阵')
    sample_count, feature_count = feature_matrix.shape
    if sample_count != target_vector.size:
        raise ValueError('特征行数与目标变量长度不一致')
    if sample_count <= feature_count + 1:
        raise ValueError('样本数不足：至少应大于特征数加 1')
    if not np.isfinite(feature_matrix).all():
        raise ValueError('特征中包含 NaN 或无穷大，请先清洗数据')
    if not np.isfinite(target_vector).all():
        raise ValueError('目标变量中包含 NaN 或无穷大，请先清洗数据')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence 必须位于 0 和 1 之间')

    if feature_names is None:
        feature_names = [f'x{j + 1}' for j in range(feature_count)]
    if len(feature_names) != feature_count:
        raise ValueError('feature_names 的数量必须与特征列数一致')

    design_matrix = add_intercept(feature_matrix)
    parameter_count = design_matrix.shape[1]

    # 设计矩阵必须满列秩。若不满秩，说明存在完全共线性或重复变量。
    design_rank = np.linalg.matrix_rank(design_matrix)
    if design_rank < parameter_count:
        raise ValueError(
            '设计矩阵不满秩，可能存在完全共线性的变量，请删除或合并相关特征'
        )

    # OLS 系数：beta = argmin ||X beta - y||²。
    coefficients, *_ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)
    fitted_values = design_matrix @ coefficients
    residuals = target_vector - fitted_values

    # 平方和分解：SST = SSR + SSE。
    residual_sum_squares = np.sum(residuals ** 2)
    total_sum_squares = np.sum((target_vector - target_vector.mean()) ** 2)
    regression_sum_squares = total_sum_squares - residual_sum_squares

    degrees_of_freedom_model = parameter_count - 1
    degrees_of_freedom_residual = sample_count - parameter_count

    # R² 表示已解释的变异比例；调整 R² 会惩罚无用的解释变量。
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    adjusted_r_squared = 1.0 - (
        (1.0 - r_squared) * (sample_count - 1) / degrees_of_freedom_residual
    )

    mean_squared_error = residual_sum_squares / degrees_of_freedom_residual
    root_mean_squared_error = np.sqrt(mean_squared_error)
    mean_absolute_error = np.mean(np.abs(residuals))

    # 回归均方除以残差均方，构造整体显著性 F 检验。
    f_statistic = (
        regression_sum_squares / degrees_of_freedom_model
    ) / mean_squared_error
    f_p_value = stats.f.sf(
        f_statistic, degrees_of_freedom_model, degrees_of_freedom_residual
    )

    # 系数协方差矩阵为 sigma² * (X'X)^(-1)。
    covariance_matrix = mean_squared_error * np.linalg.inv(
        design_matrix.T @ design_matrix
    )
    standard_errors = np.sqrt(np.diag(covariance_matrix))
    t_statistics = coefficients / standard_errors
    p_values = 2.0 * stats.t.sf(
        np.abs(t_statistics), degrees_of_freedom_residual
    )

    # 置信区间使用残差自由度对应的 t 分位数，而不是标准正态分位数。
    alpha = 1.0 - confidence
    t_critical = stats.t.ppf(
        1.0 - alpha / 2.0, degrees_of_freedom_residual
    )
    interval_margin = t_critical * standard_errors

    parameter_names = ['截距'] + list(feature_names)
    coefficient_table = pd.DataFrame({
        '变量': parameter_names,
        '系数': coefficients,
        '标准误': standard_errors,
        't值': t_statistics,
        'p值': p_values,
        f'{confidence:.0%}置信区间下限': coefficients - interval_margin,
        f'{confidence:.0%}置信区间上限': coefficients + interval_margin,
    })

    # Durbin-Watson 约为 2 时，残差序列没有明显的一阶自相关。
    durbin_watson = np.sum(np.diff(residuals) ** 2) / residual_sum_squares

    return {
        'coefficients': coefficients,
        'coefficient_table': coefficient_table,
        'feature_names': list(feature_names),
        'fitted_values': fitted_values,
        'residuals': residuals,
        'r_squared': r_squared,
        'adjusted_r_squared': adjusted_r_squared,
        'rmse': root_mean_squared_error,
        'mae': mean_absolute_error,
        'f_statistic': f_statistic,
        'f_p_value': f_p_value,
        'durbin_watson': durbin_watson,
        'condition_number': np.linalg.cond(design_matrix),
        'vif_table': calculate_vif(feature_matrix, feature_names),
        'confidence': confidence,
        'degrees_of_freedom_residual': degrees_of_freedom_residual,
    }


def predict_multiple_linear_regression(model, new_features):
    """使用已经拟合的模型预测新样本。"""
    new_matrix = np.asarray(new_features, dtype=float)
    if new_matrix.ndim == 1:
        new_matrix = new_matrix.reshape(1, -1)
    if new_matrix.ndim != 2:
        raise ValueError('new_features 必须是二维矩阵')
    if new_matrix.shape[1] != len(model['feature_names']):
        raise ValueError('新样本的特征数量必须与训练时一致')
    if not np.isfinite(new_matrix).all():
        raise ValueError('新样本中包含 NaN 或无穷大')

    return add_intercept(new_matrix) @ model['coefficients']

# Test_CTA_trend_Robustness.py
# ------------------------------------------------------------
# 基于 Test_CTA_trend_matrix.py 的三个趋势因子，构建“鲁棒性验证”一揽子脚本：
# (1) Vol-adjusted momentum:  R_k / sigma
# (2) t-stat momentum (basic / Newey-West HAC)
# (3) z-score + tanh smoothing: tanh((x - rolling_mean)/rolling_std)
#
# 鲁棒性检验覆盖：
#   A. 参数平坦性（parameter surface）：lookback / vol_lookback / NW lags / z窗口 / tanh强度
#   B. 子样本一致性（subsample）：前后半样本 + 高/低波动样本
#   C. 现实摩擦敏感性（frictions）：手续费敏感性 + 信号滞后执行
#   D. 截面鲁棒性（leave-k-out）：随机剔除部分资产后重复回测
#
# 输出：
#   - ./robustness_outputs/<asset_name>/ 下的 CSV 汇总表 & 热力图/曲线图
#
# 依赖：
#   - 你现有 Backtest.py（提供 Back_test / position_date / rebase_nav 等）
#   - 价格数据: <asset_name>.csv （index=日期, columns=资产；第1列可作为benchmark）
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from numpy.lib.stride_tricks import sliding_window_view

# 复用你原工程的回测工具
from Backtest import *  # noqa

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

'''##################################################################################################################'''


# =============================
# 信号计算（沿用/小幅增强）
# =============================

def _to_returns(prices: pd.DataFrame, method: str = "pct") -> pd.DataFrame:
    if method == "log":
        return np.log(prices).diff()
    return prices.pct_change()


def clip_signal(x: pd.DataFrame | pd.Series, lo: float = -10.0, hi: float = 10.0):
    return x.clip(lower=lo, upper=hi)


def compute_vol_adj_mom(
        prices: pd.DataFrame,
        lookback: int = 126,
        vol_lookback: int = 63,
        returns_method: str = "pct",
        annualize_vol: bool = True,
        clip: tuple[float, float] | None = (-10.0, 10.0),
) -> pd.DataFrame:
    ret = _to_returns(prices, method=returns_method)
    cum_ret = prices / prices.shift(lookback) - 1.0

    vol = ret.rolling(vol_lookback).std()
    if annualize_vol:
        vol = vol * np.sqrt(252)

    sig = cum_ret / vol
    if clip is not None:
        sig = clip_signal(sig, clip[0], clip[1])
    return sig.dropna()


def compute_t_stat_basic(
        prices: pd.DataFrame,
        lookback: int = 126,
        returns_method: str = "pct",
        clip: tuple[float, float] | None = (-10.0, 10.0),
) -> pd.DataFrame:
    ret = _to_returns(prices, method=returns_method)
    mean_r = ret.rolling(lookback).mean()
    std_r = ret.rolling(lookback).std()
    stderr = std_r / np.sqrt(lookback)
    sig = mean_r / stderr
    if clip is not None:
        sig = clip_signal(sig, clip[0], clip[1])
    return sig.dropna()


def compute_t_stat_nw(
        prices: pd.DataFrame,
        lookback: int = 126,
        lags: int = 5,
        returns_method: str = "pct",
        ddof: int = 0,
        clip: tuple[float, float] | None = (-10.0, 10.0),
) -> pd.DataFrame:
    if lags < 0:
        raise ValueError("lags must be >= 0")
    if lookback <= 1:
        raise ValueError("lookback must be > 1")

    ret = _to_returns(prices, method=returns_method)
    out = pd.DataFrame(index=ret.index, columns=ret.columns, dtype="float64")

    weights = np.array([], dtype="float64") if lags == 0 else (1 - (np.arange(1, lags + 1) / (lags + 1)))

    for col in ret.columns:
        r = ret[col].to_numpy(dtype="float64")
        if len(r) < lookback:
            continue

        w = sliding_window_view(r, window_shape=lookback)
        valid = np.isfinite(w).all(axis=1)
        if not np.any(valid):
            continue

        wv = w[valid]
        mu = wv.mean(axis=1)
        x = wv - mu[:, None]

        n = lookback
        denom0 = (n - ddof)
        gamma0 = (x * x).sum(axis=1) / denom0

        lrv = gamma0.copy()
        for l in range(1, lags + 1):
            cov_l = (x[:, l:] * x[:, :-l]).sum(axis=1) / denom0
            lrv += 2.0 * weights[l - 1] * cov_l

        se_mean = np.sqrt(lrv / n)
        tval = np.where(se_mean > 0, mu / se_mean, np.nan)

        t_full = np.full(w.shape[0], np.nan, dtype="float64")
        t_full[valid] = tval

        end_pos = np.arange(lookback - 1, len(r))
        out.iloc[end_pos, out.columns.get_loc(col)] = t_full

    if clip is not None:
        out = clip_signal(out, clip[0], clip[1])
    return out.dropna()


# -----------------------------
# (3) z-score + tanh smoothing (matrix)
# Signal = tanh( (x - rolling_mean)/rolling_std )
# -----------------------------
def compute_tanh_zscore(
        x: pd.DataFrame,
        z_lookback: int = 252,
        min_periods: int | None = None,
        ddof: int = 0,
        clip_z: float | None = 8.0,
        tanh_k: float = 1.0,
) -> pd.DataFrame:
    """对任意输入 x 做 rolling z-score，再 tanh 压缩。"""
    if min_periods is None:
        min_periods = z_lookback

    mu = x.rolling(z_lookback, min_periods=min_periods).mean()
    sd = x.rolling(z_lookback, min_periods=min_periods).std(ddof=ddof).replace(0.0, np.nan)

    z = (x - mu) / sd
    if clip_z is not None:
        z = z.clip(lower=-clip_z, upper=clip_z)

    out = np.tanh(tanh_k * z)
    return pd.DataFrame(out, index=x.index, columns=x.columns).dropna()


"""
(3) tanh_zscore因子中的 x 定义：最朴素的时间序列动量
这里不单独封装为函数：mom = prices / prices.shift(mom_lookback) - 1.0
随后交给 compute_tanh_zscore(...) 做 rolling z-score + tanh。
"""

'''##################################################################################################################'''


# =============================
# 持仓构建（支持信号滞后）
# =============================

def weight_longshort(
        factor_df: pd.DataFrame,
        hold_dates,
        top_n: int = 3,
        bottom_n: int = 3,
) -> pd.DataFrame:
    factor_df = factor_df.sort_index()
    rows = []

    for d in hold_dates:
        d = pd.to_datetime(d)
        if d not in factor_df.index:
            continue

        x = factor_df.loc[d].dropna()
        min_required = max(top_n, 0) + max(bottom_n, 0)
        if len(x) < min_required or min_required == 0:
            continue

        x_sorted = x.sort_values(ascending=False)

        if top_n > 0:
            longs = x_sorted.head(top_n).index
            w_long = 1.0 / top_n
            for code in longs:
                rows.append({"Date": d, "S_INFO_WINDCODE": code, "Weight": w_long})

        if bottom_n > 0:
            shorts = x_sorted.tail(bottom_n).index
            w_short = -1.0 / bottom_n
            for code in shorts:
                rows.append({"Date": d, "S_INFO_WINDCODE": code, "Weight": w_short})

    Hold_df = pd.DataFrame(rows)
    if not Hold_df.empty:
        Hold_df = Hold_df.sort_values(["Date", "S_INFO_WINDCODE"]).reset_index(drop=True)

    return Hold_df


def weight_longshort_with_lag(
        factor_df: pd.DataFrame,
        hold_dates,
        signal_lag: int = 0,
        top_n: int = 3,
        bottom_n: int = 3,
) -> pd.DataFrame:
    """signal_lag=1 表示调仓日用前一交易日信号（更贴近可交易性）。"""
    factor_df = factor_df.sort_index()
    idx = factor_df.index

    rows = []
    for d in hold_dates:
        d = pd.to_datetime(d)
        if d not in idx:
            # 若调仓日不是交易日，跳过
            continue

        pos = idx.get_loc(d)
        pos_sig = pos - signal_lag
        if pos_sig < 0:
            continue
        d_sig = idx[pos_sig]

        x = factor_df.loc[d_sig].dropna()
        min_required = max(top_n, 0) + max(bottom_n, 0)
        if len(x) < min_required or min_required == 0:
            continue

        x_sorted = x.sort_values(ascending=False)

        if top_n > 0:
            longs = x_sorted.head(top_n).index
            w_long = 1.0 / top_n
            for code in longs:
                rows.append({"Date": d, "S_INFO_WINDCODE": code, "Weight": w_long})

        if bottom_n > 0:
            shorts = x_sorted.tail(bottom_n).index
            w_short = -1.0 / bottom_n
            for code in shorts:
                rows.append({"Date": d, "S_INFO_WINDCODE": code, "Weight": w_short})

    Hold_df = pd.DataFrame(rows)
    if not Hold_df.empty:
        Hold_df = Hold_df.sort_values(["Date", "S_INFO_WINDCODE"]).reset_index(drop=True)

    return Hold_df


'''##################################################################################################################'''


# =============================
# 指标与工具
# =============================

def _max_drawdown(nav: pd.Series) -> float:
    x = nav.values.astype(float)
    peak = np.maximum.accumulate(x)
    dd = x / peak - 1.0
    return float(dd.min())


def _perf_stats(nav_df: pd.DataFrame, freq: int = 252) -> Dict[str, float]:
    nav = nav_df["Nav"].astype(float).copy()
    ret = nav.pct_change().dropna()
    if ret.empty:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "mdd": np.nan}

    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (freq / len(ret)) - 1
    ann_vol = ret.std() * math.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    mdd = _max_drawdown(nav)
    return {"ann_ret": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe), "mdd": float(mdd)}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _bench_from_prices(price_table: pd.DataFrame) -> pd.DataFrame:
    bench = (1 + price_table.iloc[:, 0].pct_change().fillna(0)).cumprod().to_frame("Nav")
    return bench


def _align_rebase(nav: pd.DataFrame, bench: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_idx = bench.index.intersection(nav.index).sort_values()
    nav = nav.loc[common_idx].ffill()
    bench = bench.loc[common_idx].ffill()
    nav = rebase_nav(nav)
    bench = rebase_nav(bench)
    return nav, bench


'''##################################################################################################################'''


# =============================
# 鲁棒性测试主体
# =============================

@dataclass
class RunConfig:
    asset_name: str = "有色"
    position_frequency: str = "week"  # 'day'/'week'/'month' 由你的 position_date 决定
    top_n: int = 1
    bottom_n: int = 1
    base_fee: float = 0.0003
    seed: int = 42


def run_one_backtest(
        price_table: pd.DataFrame,
        hold_dates,
        signal: pd.DataFrame,
        top_n: int,
        bottom_n: int,
        fee: float,
        signal_lag: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    hold_df = weight_longshort_with_lag(signal, hold_dates, signal_lag=signal_lag, top_n=top_n, bottom_n=bottom_n)
    nav = Back_test(hold_df, price_table, vfee=fee)
    bench = _bench_from_prices(price_table)
    nav, bench = _align_rebase(nav, bench)
    stats = _perf_stats(nav)
    return nav, stats


def subsample_masks(price_table: pd.DataFrame) -> Dict[str, pd.Index]:
    """返回每个子样本的日期索引（在回测对齐后取交集）。"""
    idx = price_table.index
    mid = idx[len(idx) // 2]

    # 用benchmark收益的滚动波动做高/低波动划分
    bench = _bench_from_prices(price_table)["Nav"]
    bench_ret = bench.pct_change().fillna(0)
    rv = bench_ret.rolling(63).std() * np.sqrt(252)
    q_low = rv.quantile(0.3)
    q_high = rv.quantile(0.7)

    masks = {
        "full": idx,
        "first_half": idx[idx <= mid],
        "second_half": idx[idx > mid],
        "low_vol": idx[rv <= q_low],
        "high_vol": idx[rv >= q_high],
    }
    return masks


def apply_subsample(nav: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    common = nav.index.intersection(dates)
    return nav.loc[common].copy()


def robustness_parameter_surface(
        price_table: pd.DataFrame,
        hold_dates,
        outdir: str,
        factor_name: str,
        signal_builder: Callable[..., pd.DataFrame],
        grid: List[Dict],
        top_n: int,
        bottom_n: int,
        fee: float,
        signal_lag: int = 0,
) -> pd.DataFrame:
    rows = []
    for g in grid:
        sig = signal_builder(**g)
        nav, stats = run_one_backtest(price_table, hold_dates, sig, top_n, bottom_n, fee, signal_lag=signal_lag)
        row = {**g, **stats}
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, f"{factor_name}_parameter_surface.csv"), index=False)
    return res


def plot_heatmap(df: pd.DataFrame, x: str, y: str, v: str, title: str, outpath: str):
    if df.empty:
        return
    piv = df.pivot_table(index=y, columns=x, values=v, aggfunc="mean")
    plt.figure(figsize=(10, 6))
    plt.imshow(piv.values, aspect='auto')
    plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns], rotation=45, ha='right')
    plt.yticks(range(len(piv.index)), [str(i) for i in piv.index])
    plt.colorbar(label=v)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def robustness_subsample(
        price_table: pd.DataFrame,
        hold_dates,
        outdir: str,
        factor_name: str,
        signal: pd.DataFrame,
        top_n: int,
        bottom_n: int,
        fee: float,
        signal_lag: int = 0,
) -> pd.DataFrame:
    nav_full, _ = run_one_backtest(price_table, hold_dates, signal, top_n, bottom_n, fee, signal_lag=signal_lag)
    masks = subsample_masks(price_table)

    rows = []
    for k, dates in masks.items():
        nav_k = apply_subsample(nav_full, dates)
        stats = _perf_stats(nav_k)
        rows.append({"subsample": k, **stats})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, f"{factor_name}_subsample.csv"), index=False)
    return res


def robustness_frictions(
        price_table: pd.DataFrame,
        hold_dates,
        outdir: str,
        factor_name: str,
        signal: pd.DataFrame,
        top_n: int,
        bottom_n: int,
        fees: List[float],
        lags: List[int],
) -> pd.DataFrame:
    rows = []
    for fee in fees:
        for lag in lags:
            _, stats = run_one_backtest(price_table, hold_dates, signal, top_n, bottom_n, fee, signal_lag=lag)
            rows.append({"fee": fee, "signal_lag": lag, **stats})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, f"{factor_name}_frictions.csv"), index=False)
    return res


def robustness_leave_k_out(
        price_table: pd.DataFrame,
        hold_dates,
        outdir: str,
        factor_name: str,
        signal_builder_on_prices: Callable[[pd.DataFrame], pd.DataFrame],
        top_n: int,
        bottom_n: int,
        fee: float,
        signal_lag: int = 0,
        n_iter: int = 50,
        drop_frac: float = 0.2,
        seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets = list(price_table.columns)
    if len(assets) <= 2:
        # 资产太少就没法做 leave-k-out
        return pd.DataFrame()

    rows = []
    for i in range(n_iter):
        k = max(1, int(len(assets) * drop_frac))
        drop = set(rng.choice(assets, size=k, replace=False).tolist())
        keep_cols = [c for c in assets if c not in drop]
        pt = price_table[keep_cols].copy()

        sig = signal_builder_on_prices(pt)
        _, stats = run_one_backtest(pt, hold_dates, sig, top_n, bottom_n, fee, signal_lag=signal_lag)
        rows.append({"iter": i, "dropped": ",".join(sorted(drop)), "n_keep": len(keep_cols), **stats})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, f"{factor_name}_leave_k_out.csv"), index=False)
    return res


def plot_nav_compare(bench: pd.DataFrame, nav: pd.DataFrame, title: str, outpath: str):
    plt.figure(figsize=(13, 7))
    plt.plot(bench.index, bench["Nav"].values, label='基准')
    plt.plot(nav.index, nav["Nav"].values, label='策略净值')
    plt.xlabel('Date')
    plt.ylabel('Net Asset Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# =============================
# 主程序：定义三因子的“默认参数 + 鲁棒性计划”
# =============================

def main(cfg: RunConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    price_table = pd.read_csv(f"{cfg.asset_name}.csv", index_col=0, parse_dates=True)
    price_table = price_table.sort_index()

    hold_dates = position_date(price_table, vPeriod=cfg.position_frequency)

    outdir = _ensure_dir(os.path.join("robustness_outputs", cfg.asset_name))

    # -----------------------------
    # 统一 benchmark
    # -----------------------------
    bench = _bench_from_prices(price_table)

    # -----------------------------
    # (1) vol-adjusted momentum
    # -----------------------------
    f1_name = "vol_adj_mom"
    f1_default = dict(lookback=126, vol_lookback=63)

    def f1_builder(**kwargs):
        return compute_vol_adj_mom(price_table, **kwargs)

    sig1 = f1_builder(**f1_default)
    nav1, _ = run_one_backtest(price_table, hold_dates, sig1, cfg.top_n, cfg.bottom_n, cfg.base_fee, signal_lag=0)
    nav1, bench1 = _align_rebase(nav1, bench)
    plot_nav_compare(bench1, nav1, f"{cfg.asset_name} | {f1_name} | default",
                     os.path.join(outdir, f"{f1_name}_default_nav.png"))

    grid1 = []
    for lb in [63, 126, 252]:
        for vb in [21, 63, 126]:
            grid1.append({"lookback": lb, "vol_lookback": vb})

    res1 = robustness_parameter_surface(
        price_table, hold_dates, outdir, f1_name,
        signal_builder=lambda lookback, vol_lookback: compute_vol_adj_mom(price_table, lookback=lookback,
                                                                          vol_lookback=vol_lookback),
        grid=grid1,
        top_n=cfg.top_n, bottom_n=cfg.bottom_n, fee=cfg.base_fee, signal_lag=0,
    )
    plot_heatmap(res1, x="lookback", y="vol_lookback", v="sharpe",
                 title=f"{cfg.asset_name} | {f1_name} Sharpe heatmap",
                 outpath=os.path.join(outdir, f"{f1_name}_heatmap_sharpe.png"))

    robustness_subsample(price_table, hold_dates, outdir, f1_name, sig1, cfg.top_n, cfg.bottom_n, cfg.base_fee)
    robustness_frictions(price_table, hold_dates, outdir, f1_name, sig1, cfg.top_n, cfg.bottom_n,
                         fees=[0.0, cfg.base_fee, 0.0010], lags=[0, 1])
    robustness_leave_k_out(
        price_table, hold_dates, outdir, f1_name,
        signal_builder_on_prices=lambda pt: compute_vol_adj_mom(pt, **f1_default),
        top_n=cfg.top_n, bottom_n=cfg.bottom_n, fee=cfg.base_fee, signal_lag=0,
        n_iter=50, drop_frac=0.2, seed=cfg.seed,
    )

    # -----------------------------
    # (2) t-stat momentum (NW)
    # -----------------------------
    f2_name = "tstat_nw"
    f2_default = dict(lookback=126, lags=5)

    def f2_builder(**kwargs):
        return compute_t_stat_nw(price_table, **kwargs)

    sig2 = f2_builder(**f2_default)
    nav2, _ = run_one_backtest(price_table, hold_dates, sig2, cfg.top_n, cfg.bottom_n, cfg.base_fee, signal_lag=0)
    nav2, bench2 = _align_rebase(nav2, bench)
    plot_nav_compare(bench2, nav2, f"{cfg.asset_name} | {f2_name} | default",
                     os.path.join(outdir, f"{f2_name}_default_nav.png"))

    grid2 = []
    for lb in [26, 63, 126, 252]:
        for L in [0, 1, 3, 5, 10]:
            grid2.append({"lookback": lb, "lags": L})

    res2 = robustness_parameter_surface(
        price_table, hold_dates, outdir, f2_name,
        signal_builder=lambda lookback, lags: compute_t_stat_nw(price_table, lookback=lookback, lags=lags),
        grid=grid2,
        top_n=cfg.top_n, bottom_n=cfg.bottom_n, fee=cfg.base_fee, signal_lag=0,
    )
    plot_heatmap(res2, x="lookback", y="lags", v="sharpe",
                 title=f"{cfg.asset_name} | {f2_name} Sharpe heatmap",
                 outpath=os.path.join(outdir, f"{f2_name}_heatmap_sharpe.png"))

    robustness_subsample(price_table, hold_dates, outdir, f2_name, sig2, cfg.top_n, cfg.bottom_n, cfg.base_fee)
    robustness_frictions(price_table, hold_dates, outdir, f2_name, sig2, cfg.top_n, cfg.bottom_n,
                         fees=[0.0, cfg.base_fee, 0.0010], lags=[0, 1])
    robustness_leave_k_out(
        price_table, hold_dates, outdir, f2_name,
        signal_builder_on_prices=lambda pt: compute_t_stat_nw(pt, **f2_default),
        top_n=cfg.top_n, bottom_n=cfg.bottom_n, fee=cfg.base_fee, signal_lag=0,
        n_iter=50, drop_frac=0.2, seed=cfg.seed,
    )

    # -----------------------------
    # (3) z-score + tanh smoothing（建议：对动量做 zscore+tanh）
    # -----------------------------
    f3_name = "tanh_mom_z"
    f3_default = dict(mom_lookback=126, z_lookback=252, clip_z=8.0, tanh_k=1.0)

    def f3_builder(**kwargs):
        mom_lookback = kwargs.get("mom_lookback", 126)
        mom = price_table / price_table.shift(mom_lookback) - 1.0
        return compute_tanh_zscore(mom, z_lookback=kwargs.get("z_lookback", 252), clip_z=kwargs.get("clip_z", 8.0),
                                   tanh_k=kwargs.get("tanh_k", 1.0))

    sig3 = f3_builder(**f3_default)
    # 这里给一个稍宽的 top/bottom，因为 tanh 信号通常更平滑
    nav3, _ = run_one_backtest(price_table, hold_dates, sig3, max(cfg.top_n, 3), max(cfg.bottom_n, 3), cfg.base_fee,
                               signal_lag=0)
    nav3, bench3 = _align_rebase(nav3, bench)
    plot_nav_compare(bench3, nav3, f"{cfg.asset_name} | {f3_name} | default",
                     os.path.join(outdir, f"{f3_name}_default_nav.png"))

    grid3 = []
    for mlb in [63, 126, 252]:
        for zlb in [126, 252]:
            for k in [0.5, 1.0, 2.0]:
                grid3.append({"mom_lookback": mlb, "z_lookback": zlb, "tanh_k": k})

    res3 = robustness_parameter_surface(
        price_table, hold_dates, outdir, f3_name,
        signal_builder=lambda mom_lookback, z_lookback, tanh_k: compute_tanh_zscore(
            price_table / price_table.shift(mom_lookback) - 1.0,
            z_lookback=z_lookback,
            clip_z=8.0,
            tanh_k=tanh_k,
        ),
        grid=grid3,
        top_n=max(cfg.top_n, 3), bottom_n=max(cfg.bottom_n, 3), fee=cfg.base_fee, signal_lag=0,
    )
    # 这张热力图只画一个切片：固定 tanh_k=1
    res3_k1 = res3[res3.get("tanh_k") == 1.0].copy() if not res3.empty else res3
    if not res3_k1.empty:
        plot_heatmap(res3_k1, x="mom_lookback", y="z_lookback", v="sharpe",
                     title=f"{cfg.asset_name} | {f3_name} Sharpe heatmap (tanh_k=1)",
                     outpath=os.path.join(outdir, f"{f3_name}_heatmap_sharpe_k1.png"))

    robustness_subsample(price_table, hold_dates, outdir, f3_name, sig3, max(cfg.top_n, 3), max(cfg.bottom_n, 3),
                         cfg.base_fee)
    robustness_frictions(price_table, hold_dates, outdir, f3_name, sig3, max(cfg.top_n, 3), max(cfg.bottom_n, 3),
                         fees=[0.0, cfg.base_fee, 0.0010], lags=[0, 1])
    robustness_leave_k_out(
        price_table, hold_dates, outdir, f3_name,
        signal_builder_on_prices=lambda pt: compute_tanh_zscore(
            pt / pt.shift(f3_default["mom_lookback"]) - 1.0,
            z_lookback=f3_default["z_lookback"],
            clip_z=8.0,
            tanh_k=f3_default["tanh_k"],
        ),
        top_n=max(cfg.top_n, 3), bottom_n=max(cfg.bottom_n, 3), fee=cfg.base_fee, signal_lag=0,
        n_iter=50, drop_frac=0.2, seed=cfg.seed,
    )

    # -----------------------------
    # 汇总（把 3 个因子默认参数结果合在一张表）
    # -----------------------------
    summary_rows = []
    for name, sig, tn, bn in [
        (f1_name, sig1, cfg.top_n, cfg.bottom_n),
        (f2_name, sig2, cfg.top_n, cfg.bottom_n),
        (f3_name, sig3, max(cfg.top_n, 3), max(cfg.bottom_n, 3)),
    ]:
        nav, stats = run_one_backtest(price_table, hold_dates, sig, tn, bn, cfg.base_fee, signal_lag=0)
        summary_rows.append(
            {"factor": name, "top_n": tn, "bottom_n": bn, "fee": cfg.base_fee, "signal_lag": 0, **stats})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(outdir, "summary_default.csv"), index=False)

    print(f"Done. Outputs saved to: {outdir}")


if __name__ == "__main__":
    cfg = RunConfig(
        asset_name="化工",
        position_frequency="week",
        top_n=1,
        bottom_n=1,
        base_fee=0.0003,
        seed=42,
    )
    main(cfg)

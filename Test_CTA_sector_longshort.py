#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze chemical sector strategy performance and robustness.

Pure-Python (no pandas/numpy) implementation for the repo's daily update data.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


DATA_PATH = "日频更新数据/日频趋势化工板块数据.csv"
REPORT_PATH = "chem_sector_report_long_short.md"




@dataclass
class PerfStats:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    total_return: float
    n_days: int


def read_price_csv(path: str) -> Tuple[List[str], Dict[str, List[Optional[float]]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = header[1:]
        data: Dict[str, List[Optional[float]]] = {c: [] for c in cols}
        dates: List[str] = []
        for row in reader:
            if not row:
                continue
            dates.append(row[0])
            for i, col in enumerate(cols, start=1):
                val = row[i].strip() if i < len(row) else ""
                if val == "":
                    data[col].append(None)
                else:
                    try:
                        data[col].append(float(val))
                    except ValueError:
                        data[col].append(None)
    return dates, data


def forward_fill(series: List[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    last: Optional[float] = None
    for v in series:
        if v is None:
            out.append(last)
        else:
            last = v
            out.append(v)
    return out


def compute_returns(prices: List[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = [None]
    for i in range(1, len(prices)):
        p0 = prices[i - 1]
        p1 = prices[i]
        if p0 is None or p1 is None or p0 == 0:
            out.append(None)
        else:
            out.append(p1 / p0 - 1.0)
    return out


def compute_vol_adj_mom_scores(
    prices: List[Optional[float]],
    lookback: int,
    vol_lookback: int = 20,
    annualize_vol: bool = True,
    clip: Tuple[float, float] | None = (-10.0, 10.0),
) -> List[Optional[float]]:
    scores: List[Optional[float]] = [None] * len(prices)
    returns = compute_returns(prices)
    for i in range(lookback, len(prices)):
        p0 = prices[i - lookback]
        p1 = prices[i]
        if p0 is None or p1 is None or p0 == 0:
            scores[i] = None
            continue
        cum_ret = p1 / p0 - 1.0

        if i < vol_lookback:
            scores[i] = None
            continue
        window = returns[i - vol_lookback + 1 : i + 1]
        if any(r is None for r in window):
            scores[i] = None
            continue
        mean = sum(window) / vol_lookback
        var = sum((r - mean) ** 2 for r in window) / max(vol_lookback - 1, 1)
        vol = math.sqrt(var)
        if annualize_vol:
            vol *= math.sqrt(252.0)
        if vol == 0:
            scores[i] = None
            continue
        val = cum_ret / vol
        if clip is not None:
            val = min(max(val, clip[0]), clip[1])
        scores[i] = val
    return scores


def compute_nw_tstat_scores(
    returns: List[Optional[float]],
    lookback: int,
    lags: int,
    ddof: int = 0,
    clip: Tuple[float, float] | None = (-10.0, 10.0),
) -> List[Optional[float]]:
    scores: List[Optional[float]] = [None] * len(returns)
    if lookback <= 1:
        return scores
    if lags < 0:
        lags = 0

    for i in range(lookback, len(returns)):
        window = returns[i - lookback + 1 : i + 1]
        if any(r is None for r in window):
            scores[i] = None
            continue

        mean = sum(window) / lookback
        demeaned = [r - mean for r in window]
        denom0 = lookback - ddof
        gamma0 = sum(x * x for x in demeaned) / denom0

        lrv = gamma0
        for lag in range(1, lags + 1):
            cov = 0.0
            for t in range(lag, lookback):
                cov += demeaned[t] * demeaned[t - lag]
            cov /= denom0
            weight = 1.0 - lag / (lags + 1.0)
            lrv += 2.0 * weight * cov

        if lrv <= 0:
            scores[i] = None
            continue

        se_mean = math.sqrt(lrv / lookback)
        score = mean / se_mean if se_mean > 0 else None
        if score is not None and clip is not None:
            score = min(max(score, clip[0]), clip[1])
        scores[i] = score
    return scores


def compute_tanh_scores(
    prices: List[Optional[float]],
    mom_lookback: int,
    z_lookback: int = 252,
    ddof: int = 0,
    clip_z: float | None = 8.0,
    tanh_k: float = 1.0,
) -> List[Optional[float]]:
    scores: List[Optional[float]] = [None] * len(prices)
    if mom_lookback <= 1:
        return scores

    mom_series: List[Optional[float]] = [None] * len(prices)
    for i in range(mom_lookback, len(prices)):
        p0 = prices[i - mom_lookback]
        p1 = prices[i]
        if p0 is None or p1 is None or p0 == 0:
            mom_series[i] = None
        else:
            mom_series[i] = p1 / p0 - 1.0

    if z_lookback <= 1:
        return scores

    sum_mom = 0.0
    sumsq_mom = 0.0
    invalid = 0

    for i, val in enumerate(mom_series):
        if val is None:
            invalid += 1
        else:
            sum_mom += val
            sumsq_mom += val * val

        if i >= z_lookback:
            old = mom_series[i - z_lookback]
            if old is None:
                invalid -= 1
            else:
                sum_mom -= old
                sumsq_mom -= old * old

        if i < z_lookback - 1 or invalid > 0:
            continue

        mean = sum_mom / z_lookback
        denom = max(z_lookback - ddof, 1)
        var = (sumsq_mom - z_lookback * mean * mean) / denom
        if var < 0:
            var = 0.0
        std = math.sqrt(var)
        if std == 0:
            scores[i] = 0.0
            continue
        z = (mom_series[i] - mean) / std if mom_series[i] is not None else 0.0
        if clip_z is not None:
            z = min(max(z, -clip_z), clip_z)
        scores[i] = math.tanh(tanh_k * z)
    return scores


def strategy_nav(
    dates: List[str],
    prices_by_col: Dict[str, List[Optional[float]]],
    lookback: int,
    mode: str = "vol_adj_mom",
    nw_lags: int = 2,
    signal_lag: int = 1,
    fee: float = 0.0003,
) -> Tuple[List[str], List[float]]:
    # forward fill prices
    prices_ff = {k: forward_fill(v) for k, v in prices_by_col.items()}
    returns = {k: compute_returns(v) for k, v in prices_ff.items()}
    if mode == "vol_adj_mom":
        scores = {k: compute_vol_adj_mom_scores(v, lookback) for k, v in prices_ff.items()}
    elif mode == "tstat":
        scores = {k: compute_nw_tstat_scores(v, lookback, nw_lags) for k, v in returns.items()}
    elif mode == "tanh":
        scores = {k: compute_tanh_scores(v, lookback) for k, v in prices_ff.items()}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    nav = [1.0]
    prev_weights: Dict[str, float] = {k: 0.0 for k in prices_by_col.keys()}

    for i in range(1, len(dates)):
        # build weights from lagged signals
        lag_idx = i - signal_lag
        if lag_idx < 0:
            nav.append(nav[-1])
            continue

        ranked: List[Tuple[str, float]] = []
        for k in prices_by_col.keys():
            score = scores[k][lag_idx]
            if score is None or returns[k][i] is None:
                continue
            ranked.append((k, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        top = ranked[:1]
        bottom = ranked[-1:] if len(ranked) >= 2 else []

        weights: Dict[str, float] = {k: 0.0 for k in prices_by_col.keys()}
        for k, _ in top:
            weights[k] = 0.5
        for k, _ in bottom:
            weights[k] = -0.5

        # gross return
        gross = 0.0
        for k, w in weights.items():
            r = returns[k][i]
            if r is None:
                continue
            gross += w * r

        # turnover cost
        turnover = 0.0
        for k in weights.keys():
            turnover += abs(weights[k] - prev_weights.get(k, 0.0))
        turnover *= 0.5
        cost = turnover * fee

        nav.append(nav[-1] * (1.0 + gross - cost))
        prev_weights = weights

    return dates, nav


def max_drawdown(nav: List[float]) -> float:
    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def annualized_stats(nav: List[float]) -> PerfStats:
    if len(nav) < 2:
        return PerfStats(0.0, 0.0, 0.0, 0.0, 0.0, len(nav))
    rets = []
    for i in range(1, len(nav)):
        r = nav[i] / nav[i - 1] - 1.0
        rets.append(r)
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
    vol = math.sqrt(var)
    ann_return = (nav[-1] / nav[0]) ** (252 / len(rets)) - 1.0
    ann_vol = vol * math.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0.0
    mdd = max_drawdown(nav)
    total_ret = nav[-1] / nav[0] - 1.0
    return PerfStats(ann_return, ann_vol, sharpe, mdd, total_ret, len(rets))


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def split_into_parts(
    dates: List[str],
    nav: List[float],
    parts: int,
) -> List[Tuple[List[str], List[float]]]:
    if parts <= 1:
        return [(dates, nav)]
    n = len(nav)
    if n == 0:
        return []
    chunk = max(n // parts, 1)
    out: List[Tuple[List[str], List[float]]] = []
    start = 0
    for i in range(parts):
        end = n if i == parts - 1 else min(start + chunk, n)
        out.append((dates[start:end], nav[start:end]))
        start = end
        if start >= n:
            break
    return out


def rolling_vol(returns: List[Optional[float]], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(returns)
    for i in range(window, len(returns)):
        chunk = [r for r in returns[i - window + 1 : i + 1] if r is not None]
        if len(chunk) < window:
            out[i] = None
            continue
        mean = sum(chunk) / len(chunk)
        var = sum((r - mean) ** 2 for r in chunk) / max(len(chunk) - 1, 1)
        out[i] = math.sqrt(var)
    return out


def regime_split_by_vol(nav: List[float], window: int = 63) -> Tuple[List[float], List[float]]:
    # derive nav returns
    rets = [None]
    for i in range(1, len(nav)):
        rets.append(nav[i] / nav[i - 1] - 1.0)
    vol = rolling_vol(rets, window=window)
    # median vol (ignore None)
    valid = [v for v in vol if v is not None]
    if not valid:
        return nav, []
    valid_sorted = sorted(valid)
    mid = len(valid_sorted) // 2
    median = valid_sorted[mid]

    high_nav = [nav[0]]
    low_nav = [nav[0]]
    for i in range(1, len(nav)):
        if vol[i] is None:
            high_nav.append(high_nav[-1])
            low_nav.append(low_nav[-1])
            continue
        r = nav[i] / nav[i - 1] - 1.0
        if vol[i] >= median:
            high_nav.append(high_nav[-1] * (1.0 + r))
            low_nav.append(low_nav[-1])
        else:
            low_nav.append(low_nav[-1] * (1.0 + r))
            high_nav.append(high_nav[-1])
    return high_nav, low_nav


def summarize_stats(title: str, stats: PerfStats) -> str:
    return (
        f"| {title} | {format_pct(stats.total_return)} | {format_pct(stats.ann_return)} | "
        f"{format_pct(stats.ann_vol)} | {stats.sharpe:.2f} | {format_pct(stats.max_drawdown)} | {stats.n_days} |\n"
    )


def summarize_pair_stats(title: str, left: PerfStats, right: PerfStats) -> str:
    return (
        f"| {title} | {left.sharpe:.2f} | {right.sharpe:.2f} | "
        f"{format_pct(left.total_return)} | {format_pct(right.total_return)} |\n"
    )


def main() -> None:
    dates, prices = read_price_csv(DATA_PATH)
    # Remove benchmark column if present (use first column as benchmark maybe)
    benchmark_col = None
    if "CIFI.WI" in prices:
        benchmark_col = "CIFI.WI"
    asset_cols = [c for c in prices.keys() if c != benchmark_col]
    prices_assets = {k: prices[k] for k in asset_cols}

    lookbacks = [5, 10, 20, 63]
    lag = 1
    fee = 0.0003
    nw_lags = 2

    sections: List[str] = []
    sections.append("# 化工板块趋势策略表现与鲁棒性\n")
    sections.append(f"数据源: `{DATA_PATH}` (日频更新数据)。\n")
    sections.append("\n## 1. 基准策略设定\n")
    sections.append(
        "- 资产池：化工板块内所有品种（剔除板块指数列 `CIFI.WI`）。\n"
        "- 因子 A：`compute_vol_adj_mom`，`lookback=5`，`vol_lookback=63`，信号为动量/波动率。\n"
        "- 因子 B：`compute_t_stat_nw`，`lookback=5`, `lags=2`（HAC 修正 t-stat）。\n"
        "- 因子 C：`compute_tanh_signal`，`lookback=5`，`z_lookback=252`（raw_mom → z-score → tanh）。\n"
        "- 权重：对板块内趋势最强的品种做多（1/2），对趋势最弱的品种做空（-1/2）。\n"
        "- 执行：信号滞后 1 日执行（避免未来函数）。\n"
        "- 手续费：按日换手的 0.03% 计入。\n"
    )

    def run_section(title: str, mode: str) -> List[str]:
        out: List[str] = []
        base_dates, base_nav = strategy_nav(
            dates,
            prices_assets,
            lookback=5,
            mode=mode,
            nw_lags=nw_lags,
            signal_lag=lag,
            fee=fee,
        )
        base_stats = annualized_stats(base_nav)

        out.append(f"\n## {title}：整体表现\n")
        out.append("| 组合 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        out.append(summarize_stats("化工趋势策略", base_stats))

        out.append(f"\n## {title}：子样本稳健性（5等分）\n")
        subsamples = split_into_parts(base_dates, base_nav, parts=5)
        out.append("| 子样本 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        for idx, (_, nav) in enumerate(subsamples, start=1):
            s = annualized_stats(nav)
            out.append(summarize_stats(f"第{idx}份", s))

        out.append(f"\n## {title}：高/低波动 regime 表现\n")
        high_nav, low_nav = regime_split_by_vol(base_nav, window=63)
        s_high = annualized_stats(high_nav)
        s_low = annualized_stats(low_nav)
        out.append("| 波动区间 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        out.append(summarize_stats("高波动区间", s_high))
        out.append(summarize_stats("低波动区间", s_low))

        out.append(f"\n## {title}：参数敏感性（lookback）\n")
        out.append("| lookback | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        for lb in lookbacks:
            d, n = strategy_nav(
                dates,
                prices_assets,
                lookback=lb,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag,
                fee=fee,
            )
            s = annualized_stats(n)
            out.append(summarize_stats(str(lb), s))

        out.append(f"\n## {title}：执行滞后敏感性\n")
        out.append("| signal_lag | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        for lag_i in [0, 1, 2]:
            d, n = strategy_nav(
                dates,
                prices_assets,
                lookback=5,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag_i,
                fee=fee,
            )
            s = annualized_stats(n)
            out.append(summarize_stats(str(lag_i), s))
        return out

    sections.extend(run_section("2. vol_adj_mom 因子", "vol_adj_mom"))
    sections.extend(run_section("3. t-stat 因子（NW 调整）", "tstat"))
    sections.extend(run_section("4. tanh 因子", "tanh"))

    compare_lookbacks = [5, 10, 20]
    factor_defs = [
        ("vol_adj_mom", "vol_adj_mom"),
        ("tstat", "t-stat (NW)"),
        ("tanh", "tanh"),
    ]

    sections.append("\n## 5. 三因子在 lookback=5/10/20 的总表格\n")
    sections.append("### 5.1 基准表现（lag=1）\n")
    sections.append("| 因子 | lookback | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
    sections.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        for lb in compare_lookbacks:
            _, nav = strategy_nav(
                dates,
                prices_assets,
                lookback=lb,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag,
                fee=fee,
            )
            stats = annualized_stats(nav)
            sections.append(summarize_stats(f"{label} (lb={lb})", stats))

    sections.append("\n### 5.2 子样本稳健性（5等分：Sharpe/总收益）\n")
    sections.append("| 因子 | 子样本 | Sharpe | 总收益 |\n")
    sections.append("| --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        for lb in compare_lookbacks:
            d, nav = strategy_nav(
                dates,
                prices_assets,
                lookback=lb,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag,
                fee=fee,
            )
            subsamples = split_into_parts(d, nav, parts=5)
            for idx, (_, nav_sub) in enumerate(subsamples, start=1):
                s = annualized_stats(nav_sub)
                sections.append(
                    f"| {label} (lb={lb}) | 第{idx}份 | {s.sharpe:.2f} | {format_pct(s.total_return)} |\n"
                )

    sections.append("\n### 5.3 高/低波动 regime 表现（Sharpe/总收益）\n")
    sections.append("| 因子 | 高波动Sharpe | 低波动Sharpe | 高波动总收益 | 低波动总收益 |\n")
    sections.append("| --- | --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        for lb in compare_lookbacks:
            _, nav = strategy_nav(
                dates,
                prices_assets,
                lookback=lb,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag,
                fee=fee,
            )
            high_nav, low_nav = regime_split_by_vol(nav, window=63)
            s_high = annualized_stats(high_nav)
            s_low = annualized_stats(low_nav)
            sections.append(summarize_pair_stats(f"{label} (lb={lb})", s_high, s_low))

    sections.append("\n### 5.4 执行滞后敏感性（Sharpe, lookback=5/10/20）\n")
    sections.append("| 因子 | lookback | lag=0 Sharpe | lag=1 Sharpe | lag=2 Sharpe |\n")
    sections.append("| --- | --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        for lb in compare_lookbacks:
            sharpe_vals = []
            for lag_i in [0, 1, 2]:
                _, nav = strategy_nav(
                    dates,
                    prices_assets,
                    lookback=lb,
                    mode=mode,
                    nw_lags=nw_lags,
                    signal_lag=lag_i,
                    fee=fee,
                )
                sharpe_vals.append(annualized_stats(nav).sharpe)
            sections.append(
                f"| {label} (lb={lb}) | {lb} | {sharpe_vals[0]:.2f} | {sharpe_vals[1]:.2f} | {sharpe_vals[2]:.2f} |\n"
            )

    sections.append("\n## 6. 因子对比摘要（基准 lookback=5, lag=1）\n")
    sections.append("| 因子 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
    sections.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        _, nav = strategy_nav(
            dates,
            prices_assets,
            lookback=5,
            mode=mode,
            nw_lags=nw_lags,
            signal_lag=lag,
            fee=fee,
        )
        stats = annualized_stats(nav)
        sections.append(summarize_stats(label, stats))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("".join(sections))


if __name__ == "__main__":
    main()

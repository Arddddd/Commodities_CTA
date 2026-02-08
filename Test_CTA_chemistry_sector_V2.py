#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze chemical sector strategy performance and robustness.

Pure-Python (no pandas/numpy) implementation for the repo's daily update data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from Backtest import Back_test, position_date

DATA_PATH = "日频更新数据/日频趋势化工板块数据.csv"
REPORT_PATH = "chem_sector_report3.md"


@dataclass
class PerfStats:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    total_return: float
    n_days: int


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
    for p0, p1 in zip(prices, prices[1:]):
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
        window = returns[i - vol_lookback + 1: i + 1]
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
        window = returns[i - lookback + 1: i + 1]
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


def weight_longshort(
        factor_df: pd.DataFrame,
        hold_dates,
        top_n: int = 1,
        bottom_n: int = 1,
) -> pd.DataFrame:
    """
    输入：
      - factor_df: 日频因子表，index=日期，columns=资产代码
      - hold_dates: 调仓日列表
    输出：
      - Hold_df: [Date, S_INFO_WINDCODE, Weight]

    规则：
      - top_n = 0     → 仅空头
      - bottom_n = 0  → 仅多头
      - top_n > 0 且 bottom_n > 0 → 多空
    """
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

    hold_df = pd.DataFrame(rows)
    if not hold_df.empty:
        hold_df = hold_df.sort_values(["Date", "S_INFO_WINDCODE"]).reset_index(drop=True)

    return hold_df


def strategy_nav(
        price_table: pd.DataFrame,
        prices_by_col: Dict[str, List[Optional[float]]],
        lookback: int,
        mode: str = "vol_adj_mom",
        nw_lags: int = 2,
        signal_lag: int = 0,
        fee: float = 0.0003,
        position_frequency: str = "week",
        top_n: int = 1,
        bottom_n: int = 1,
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

    signal_df = pd.DataFrame(scores, index=price_table.index)
    if signal_lag > 0:
        signal_df = signal_df.shift(signal_lag)

    hold_dates = position_date(price_table, vPeriod=position_frequency)
    hold_df = weight_longshort(signal_df, hold_dates, top_n=top_n, bottom_n=bottom_n)
    nav_df = Back_test(hold_df, price_table, vfee=fee)
    if nav_df.empty:
        return [], []

    return nav_df.index.astype(str).tolist(), nav_df["Nav"].tolist()


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
    rets = [nav[i] / nav[i - 1] - 1.0 for i in range(1, len(nav))]
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
        chunk = [r for r in returns[i - window + 1: i + 1] if r is not None]
        if len(chunk) < window:
            out[i] = None
            continue
        mean = sum(chunk) / len(chunk)
        var = sum((r - mean) ** 2 for r in chunk) / max(len(chunk) - 1, 1)
        out[i] = math.sqrt(var)
    return out


def regime_split_by_vol(nav: List[float], window: int = 63) -> Tuple[List[float], List[float]]:
    rets = [None] + [nav[i] / nav[i - 1] - 1.0 for i in range(1, len(nav))]
    vol = rolling_vol(rets, window=window)
    valid = [v for v in vol if v is not None]
    if not valid:
        return nav, []
    median = sorted(valid)[len(valid) // 2]

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
    price_table = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    prices_for_scores = price_table.where(pd.notna(price_table), None)
    benchmark_col = "CIFI.WI" if "CIFI.WI" in prices_for_scores else None
    asset_cols = [c for c in prices_for_scores.columns if c != benchmark_col]
    prices_assets = {k: prices_for_scores[k].tolist() for k in asset_cols}

    basic_lookback = 20
    basic_vol_lookback = 20
    lookbacks = [5, 10, 20, 60]
    lag = 0
    fee = 0.0003
    nw_lags = 1
    position_frequency = "week"
    top_n = 2
    bottom_n = 0

    sections: List[str] = []
    sections.append("# 化工板块趋势策略表现与鲁棒性\n")
    sections.append(f"数据源: `{DATA_PATH}` (日频更新数据)。\n")
    sections.append(f"\n## 1. 基准策略设定（lookback={basic_lookback}）\n")
    sections.append(
        "- 资产池：化工板块内所有品种（剔除板块指数列 `CIFI.WI`）。\n"
        f"- 因子 A：`compute_vol_adj_mom`，`vol_lookback={basic_vol_lookback}`，信号为动量/波动率。\n"
        f"- 因子 B：`compute_t_stat_nw`，`lags={nw_lags}`（HAC 修正 t-stat）。\n"
        f"- 因子 C：`compute_tanh_signal`，`z_lookback=252`（raw_mom → z-score → tanh）。\n"
        f"- 权重：对板块内趋势最强的品种做多top_n={top_n}，对趋势最弱的品种做空bottom_n={bottom_n}。\n"
        f"- 调仓频率：{position_frequency}。\n"
        f"- 手续费：按日换手的 {fee * 100:.2f}% 计入。\n"
    )

    def run_section(title: str, mode: str) -> List[str]:
        out: List[str] = []
        base_dates, base_nav = strategy_nav(
            price_table,
            prices_assets,
            lookback=basic_lookback,
            mode=mode,
            nw_lags=nw_lags,
            signal_lag=lag,
            fee=fee,
            position_frequency=position_frequency,
            top_n=top_n,
            bottom_n=bottom_n,
        )
        base_stats = annualized_stats(base_nav)

        out.append(f"\n## {title}：整体表现\n")
        out.append(
            f"参数：lookback={basic_lookback}，signal_lag={lag}，fee={fee * 100:.2f}%"
            f"，frequency={position_frequency}，top_n={top_n}，bottom_n={bottom_n}"
            f"，nw_lags={nw_lags}。\n"
        )
        out.append("| 组合 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        out.append(summarize_stats("化工趋势策略", base_stats))

        out.append(f"\n## {title}：子样本稳健性（五等分）\n")
        out.append(
            f"参数：parts=5，lookback={basic_lookback}，signal_lag={lag}，fee={fee * 100:.2f}%"
            f"，frequency={position_frequency}，top_n={top_n}，bottom_n={bottom_n}"
            f"，nw_lags={nw_lags}。\n"
        )
        subsamples = split_into_parts(base_dates, base_nav, parts=5)
        out.append("| 子样本 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        for idx, (_, nav) in enumerate(subsamples, start=1):
            s = annualized_stats(nav)
            out.append(summarize_stats(f"第{idx}份", s))

        out.append(f"\n## {title}：高/低波动 regime 表现\n")
        out.append(
            f"参数：window=63，lookback={basic_lookback}，signal_lag={lag}，fee={fee * 100:.2f}%"
            f"，frequency={position_frequency}，top_n={top_n}，bottom_n={bottom_n}"
            f"，nw_lags={nw_lags}。\n"
        )
        high_nav, low_nav = regime_split_by_vol(base_nav, window=63)
        s_high = annualized_stats(high_nav)
        s_low = annualized_stats(low_nav)
        out.append("| 波动区间 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        out.append(summarize_stats("高波动区间", s_high))
        out.append(summarize_stats("低波动区间", s_low))

        out.append(f"\n## {title}：参数敏感性（lookback）\n")
        out.append(
            f"参数：lookbacks={lookbacks}，signal_lag={lag}，fee={fee * 100:.2f}%"
            f"，frequency={position_frequency}，top_n={top_n}，bottom_n={bottom_n}"
            f"，nw_lags={nw_lags}。\n"
        )
        out.append("| lookback | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
        out.append("| --- | --- | --- | --- | --- | --- | --- |\n")
        for lb in lookbacks:
            d, n = strategy_nav(
                price_table,
                prices_assets,
                lookback=lb,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag,
                fee=fee,
                position_frequency=position_frequency,
                top_n=top_n,
                bottom_n=bottom_n,
            )
            s = annualized_stats(n)
            out.append(summarize_stats(str(lb), s))

        return out

    sections.extend(run_section("2. vol_adj_mom 因子", "vol_adj_mom"))
    sections.extend(run_section("3. t-stat 因子（NW 调整）", "tstat"))
    sections.extend(run_section("4. tanh 因子", "tanh"))

    compare_lookbacks = [5, 10, 20, 60]
    factor_defs = [
        ("vol_adj_mom", "vol_adj_mom"),
        ("tstat", "t-stat (NW)"),
        ("tanh", "tanh"),
    ]

    sections.append("\n## 5. 三因子在 lookback=5/10/20/60 的总表格\n")
    sections.append(
        f"参数：lookbacks={compare_lookbacks}，signal_lag={lag}，fee={fee * 100:.2f}%"
        f"，frequency={position_frequency}，top_n={top_n}，bottom_n={bottom_n}"
        f"，nw_lags={nw_lags}。\n"
    )
    sections.append("### 5.1 基准表现\n")
    sections.append("| 因子 | lookback | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
    sections.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        for lb in compare_lookbacks:
            _, nav = strategy_nav(
                price_table,
                prices_assets,
                lookback=lb,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag,
                fee=fee,
                position_frequency=position_frequency,
                top_n=top_n,
                bottom_n=bottom_n,
            )
            stats = annualized_stats(nav)
            sections.append(summarize_stats(f"{label} (lb={lb})", stats))

    sections.append("\n### 5.2 子样本稳健性（五等分：Sharpe/总收益）\n")
    sections.append(
        f"参数：parts=5，lookbacks={compare_lookbacks}，signal_lag={lag}，fee={fee * 100:.2f}%"
        f"，frequency={position_frequency}，top_n={top_n}，bottom_n={bottom_n}"
        f"，nw_lags={nw_lags}。\n"
    )
    sections.append("| 因子 | 子样本 | Sharpe | 总收益 |\n")
    sections.append("| --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        for lb in compare_lookbacks:
            d, nav = strategy_nav(
                price_table,
                prices_assets,
                lookback=lb,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag,
                fee=fee,
                position_frequency=position_frequency,
                top_n=top_n,
                bottom_n=bottom_n,
            )
            subsamples = split_into_parts(d, nav, parts=5)
            for idx, (_, nav_sub) in enumerate(subsamples, start=1):
                s = annualized_stats(nav_sub)
                sections.append(
                    f"| {label} (lb={lb}) | 第{idx}份 | {s.sharpe:.2f} | {format_pct(s.total_return)} |\n"
                )

    sections.append("\n### 5.3 高/低波动 regime 表现（Sharpe/总收益）\n")
    sections.append(
        f"参数：window=63，lookbacks={compare_lookbacks}，signal_lag={lag}，fee={fee * 100:.2f}%"
        f"，frequency={position_frequency}，top_n={top_n}，bottom_n={bottom_n}"
        f"，nw_lags={nw_lags}。\n"
    )
    sections.append("| 因子 | 高波动Sharpe | 低波动Sharpe | 高波动总收益 | 低波动总收益 |\n")
    sections.append("| --- | --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        for lb in compare_lookbacks:
            _, nav = strategy_nav(
                price_table,
                prices_assets,
                lookback=lb,
                mode=mode,
                nw_lags=nw_lags,
                signal_lag=lag,
                fee=fee,
                position_frequency=position_frequency,
                top_n=top_n,
                bottom_n=bottom_n,
            )
            high_nav, low_nav = regime_split_by_vol(nav, window=63)
            s_high = annualized_stats(high_nav)
            s_low = annualized_stats(low_nav)
            sections.append(summarize_pair_stats(f"{label} (lb={lb})", s_high, s_low))

    sections.append(f"\n## 6. 因子对比摘要（基准 lookback={basic_lookback}, lag={lag}）\n")
    sections.append(
        f"参数：lookback=10，signal_lag={lag}，fee={fee * 100:.2f}%"
        f"，frequency={position_frequency}，top_n={top_n}，bottom_n={bottom_n}"
        f"，nw_lags={nw_lags}。\n"
    )
    sections.append("| 因子 | 总收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 样本天数 |\n")
    sections.append("| --- | --- | --- | --- | --- | --- | --- |\n")
    for mode, label in factor_defs:
        _, nav = strategy_nav(
            price_table,
            prices_assets,
            lookback=10,
            mode=mode,
            nw_lags=nw_lags,
            signal_lag=lag,
            fee=fee,
            position_frequency=position_frequency,
            top_n=top_n,
            bottom_n=bottom_n,
        )
        stats = annualized_stats(nav)
        sections.append(summarize_stats(label, stats))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("".join(sections))


if __name__ == "__main__":
    main()

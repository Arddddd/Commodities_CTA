# -*- coding: utf-8 -*-
"""Test_CTA_trend_daily_signal.py

Outputs:
- positions sheet includes `w_for_sector_vol` (the w_i used in sector vol).
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _to_returns(prices: pd.DataFrame, method: str = "pct") -> pd.DataFrame:
    if method == "log":
        return np.log(prices).diff()
    return prices.pct_change()


def strength_mapping(t: pd.Series, t0: float = 1.0, t1: float = 2.5, cap: float = 1.0) -> pd.Series:
    a = (t.abs() - t0) / max(t1 - t0, 1e-12)
    return a.clip(lower=0.0, upper=cap)


def compute_hist_vol(prices: pd.DataFrame, window: int = 252, returns_method: str = "pct",
                     annualize: bool = True) -> pd.DataFrame:
    ret = _to_returns(prices, method=returns_method)
    vol = ret.rolling(window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def compute_t_stat_nw(
        prices: pd.DataFrame,
        lookback: int = 26,
        lags: int = 5,
        returns_method: str = "pct",
        ddof: int = 0,
        clip: Tuple[float, float] | None = (-10.0, 10.0),
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
        out = out.clip(lower=clip[0], upper=clip[1])
    return out.dropna(how="all")


def sector_vol_est_annual_constcorr(weights: pd.Series, vols_annual: pd.Series, rho: float = 0.7) -> float:
    common = weights.index.intersection(vols_annual.index)
    if len(common) == 0:
        return float("nan")
    w = weights.loc[common].astype(float).to_numpy()
    s = vols_annual.loc[common].astype(float).to_numpy()

    ws = w * s
    var = float(np.sum(ws ** 2))
    if len(ws) >= 2 and rho != 0:
        pair_sum = (float(np.sum(ws)) ** 2 - float(np.sum(ws ** 2))) / 2.0
        var += 2.0 * rho * pair_sum
    return float(math.sqrt(max(var, 0.0)))


def w_for_sector_vol(raw_w: pd.Series, w_capped: pd.Series) -> pd.Series:
    """Elementwise choose the smaller absolute weight (keep sign)."""
    use_raw = raw_w.abs() <= w_capped.abs()
    out = w_capped.copy()
    out[use_raw] = raw_w[use_raw]
    return out


@dataclass
class V7Config:
    LOOKBACK: int = 26
    NW_LAGS: int = 5
    T_TH_SECTOR: float = 1.0
    T_TH_INST: float = 1.0

    TOP_K: int = 3
    N_ALTERNATES: int = 3

    TARGET_VOL_ANNUAL_PER_INST: float = 0.02
    VOL_WINDOW: int = 252
    STRENGTH_T1: float = 2.5

    PER_INST_CAP: float = 0.05
    SECTOR_VOL_CAP: float = 0.015
    RHO_INTRA_SECTOR: float = 0.7


def infer_name(describe: Optional[pd.DataFrame], code: str) -> Optional[str]:
    if describe is None:
        return None
    if "S_INFO_WINDCODE" not in describe.columns:
        return None
    hit = describe.loc[describe["S_INFO_WINDCODE"] == code]
    if hit.empty:
        return None
    for nm_col in ["S_INFO_NAME", "NAME", "品种", "中文简称"]:
        if nm_col in hit.columns:
            val = hit.iloc[0][nm_col]
            if pd.isna(val):
                return None
            return str(val)
    return None


def generate_v7(
        sector_indices: pd.DataFrame,
        sector_to_prices: Dict[str, pd.DataFrame],
        describe: Optional[pd.DataFrame],
        cfg: V7Config,
):
    sector_t = compute_t_stat_nw(sector_indices, lookback=cfg.LOOKBACK, lags=cfg.NW_LAGS)
    if sector_t.empty:
        raise ValueError("sector_t is empty: check sector_indices data length / NaNs.")
    latest_dt = sector_t.index[-1]
    sector_t_last = sector_t.loc[latest_dt].dropna()

    sector_dir = pd.Series(0, index=sector_t_last.index, dtype="int64")
    sector_dir[sector_t_last >= cfg.T_TH_SECTOR] = 1
    sector_dir[sector_t_last <= -cfg.T_TH_SECTOR] = -1

    sector_gate_df = pd.DataFrame({
        "date": latest_dt,
        "sector": sector_t_last.index,
        "sector_t": sector_t_last.values,
        "sector_dir": sector_dir.values,
        "gate_pass": (sector_dir.values != 0),
        "rule": [f"|t|>={cfg.T_TH_SECTOR} => allow" for _ in sector_t_last.index],
    })

    positions_rows: List[dict] = []
    rank_sheets: Dict[str, pd.DataFrame] = {}

    for sector, px in sector_to_prices.items():
        inst_t = compute_t_stat_nw(px, lookback=cfg.LOOKBACK, lags=cfg.NW_LAGS)
        if inst_t.empty or latest_dt not in inst_t.index:
            continue
        inst_t_last = inst_t.loc[latest_dt].dropna()
        vol_1y = compute_hist_vol(px, window=cfg.VOL_WINDOW).loc[latest_dt].dropna()

        sdir = int(sector_dir.get(sector, 0))
        st = float(sector_t_last.get(sector, np.nan))

        df_rank = pd.DataFrame({
            "date": latest_dt,
            "sector": sector,
            "wind_code": inst_t_last.index,
            "inst_t": inst_t_last.values,
        })
        df_rank["abs_t"] = df_rank["inst_t"].abs()
        df_rank["name"] = df_rank["wind_code"].apply(lambda c: infer_name(describe, c))
        df_rank["vol_1y"] = df_rank["wind_code"].map(vol_1y.to_dict())
        df_rank["sector_t"] = st
        df_rank["sector_dir"] = sdir

        def _reason(row) -> str:
            if sdir == 0:
                return "sector_gate_fail"
            tval = float(row["inst_t"])
            if sdir == 1 and tval < cfg.T_TH_INST:
                return "inst_t_below_long_threshold"
            if sdir == -1 and tval > -cfg.T_TH_INST:
                return "inst_t_above_short_threshold"
            v = row["vol_1y"]
            if not np.isfinite(v) or v <= 0:
                return "missing_or_bad_vol"
            return "pass"

        df_rank["filter_reason"] = df_rank.apply(_reason, axis=1)
        df_rank["pass"] = df_rank["filter_reason"].eq("pass")

        df_pass = df_rank[df_rank["pass"]].copy()
        if sdir == 1:
            df_pass = df_pass.sort_values(["inst_t"], ascending=False)
        elif sdir == -1:
            df_pass = df_pass.sort_values(["inst_t"], ascending=True)
        else:
            df_pass = df_pass.iloc[0:0]

        selected = df_pass.head(cfg.TOP_K).copy()
        alternates = df_pass.iloc[cfg.TOP_K: cfg.TOP_K + cfg.N_ALTERNATES].copy()

        df_rank["status"] = "filtered"
        df_rank.loc[df_rank["pass"], "status"] = "pass_not_selected"
        if not selected.empty:
            df_rank.loc[df_rank["wind_code"].isin(selected["wind_code"]), "status"] = "selected"
        if not alternates.empty:
            df_rank.loc[df_rank["wind_code"].isin(alternates["wind_code"]), "status"] = "alternate"

        sheet_name = f"sector_rank_{sector}"[:31]
        rank_sheets[sheet_name] = df_rank.sort_values(["pass", "abs_t"], ascending=[False, False])

        if selected.empty:
            continue

        t_series = selected.set_index("wind_code")["inst_t"].astype(float)
        v_series = selected.set_index("wind_code")["vol_1y"].astype(float)

        strength = strength_mapping(t_series, t0=cfg.T_TH_INST, t1=cfg.STRENGTH_T1, cap=1.0)
        raw_w = t_series * strength * (cfg.TARGET_VOL_ANNUAL_PER_INST / v_series)

        if sdir == 1:
            raw_w = raw_w.abs()
        elif sdir == -1:
            raw_w = -raw_w.abs()

        w_capped = raw_w.clip(lower=-cfg.PER_INST_CAP, upper=cfg.PER_INST_CAP)

        # NEW: use min(|raw|, |cap|) for sector vol estimation
        w_used = w_for_sector_vol(raw_w, w_capped)

        sec_vol_before = sector_vol_est_annual_constcorr(w_used, v_series, rho=cfg.RHO_INTRA_SECTOR)
        sec_scale = 1.0
        if np.isfinite(sec_vol_before) and sec_vol_before > cfg.SECTOR_VOL_CAP and sec_vol_before > 0:
            sec_scale = cfg.SECTOR_VOL_CAP / sec_vol_before

        w_final = w_capped * sec_scale

        # recompute sector vol after scaling (apply the same "min" rule consistently)
        sec_vol_after = sector_vol_est_annual_constcorr(
            w_for_sector_vol(raw_w * sec_scale, w_final),
            v_series,
            rho=cfg.RHO_INTRA_SECTOR
        )

        for code in selected["wind_code"].tolist():
            positions_rows.append({
                "date": latest_dt,
                "sector": sector,
                "wind_code": code,
                "name": infer_name(describe, code),
                "sector_t": st,
                "sector_dir": sdir,
                "inst_t": float(t_series.loc[code]),
                "vol_1y": float(v_series.loc[code]),
                "rho_intra_sector": float(cfg.RHO_INTRA_SECTOR),
                "strength": float(strength.loc[code]),
                "raw_weight": float(raw_w.loc[code]),
                "weight_capped": float(w_capped.loc[code]),
                "w_for_sector_vol": float(w_used.loc[code]),
                "sector_vol_before": float(sec_vol_before),
                "sector_scale": float(sec_scale),
                "sector_vol_after": float(sec_vol_after),
                "weight": float(w_final.loc[code]),
            })

    positions_df = pd.DataFrame(positions_rows)
    if not positions_df.empty:
        positions_df = positions_df.sort_values(["sector", "weight"], ascending=[True, False]).reset_index(drop=True)

    return positions_df, sector_gate_df, rank_sheets


def save_v7_excel(out_path: str, positions_df: pd.DataFrame, sector_gate_df: pd.DataFrame,
                  rank_sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        sector_gate_df.to_excel(writer, sheet_name="sector_gate", index=False)
        positions_df.to_excel(writer, sheet_name="positions", index=False)
        for sheet, df in rank_sheets.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)


if __name__ == "__main__":
    asset_name = '有色'
    chem = pd.read_csv(r'日频更新数据\日频趋势' + asset_name + '板块数据.csv', index_col=0, parse_dates=True)
    sector_indices = pd.DataFrame(chem.iloc[:, 0]
                                  .copy())
    sector_indices.columns = [asset_name]  # 这个列名必须与 sector_to_prices 的 key 对上
    sector_to_prices = {asset_name: chem.iloc[:, 1:].copy()}
    describe = pd.read_excel(r'南华测试.xlsx', sheet_name="描述")
    lookback = 5
    cfg = V7Config(
        LOOKBACK=lookback,  # t-stat 回看窗口
        NW_LAGS=5,  # Newey-West 滞后阶数
        T_TH_SECTOR=1.0,  # 板块 gate 阈值：|sector_t|>=1 才允许交易
        T_TH_INST=1.0,  # 品种阈值：方向一致且 |inst_t|>=1 才入候选
        TOP_K=3,  # 每板块最多持仓品种数
        N_ALTERNATES=3,  # 额外输出替补（仅解释，不交易）

        TARGET_VOL_ANNUAL_PER_INST=0.002,  # 单品种波动上限（决定 raw_weight 的“斜率”）
        VOL_WINDOW=252,  # 历史波动率回看窗口（1 Year）
        STRENGTH_T1=2.5,  # strength 映射饱和点：|t|>=2.5 -> strength=1

        PER_INST_CAP=0.05,  # 单品种仓位硬上限（权重，不是波动）
        SECTOR_VOL_CAP=0.015,  # 板块年化目标波动上限

        RHO_INTRA_SECTOR=0.7  # 板块内两两相关系数恒为 0.7
    )
    positions, sector_gate, rank_sheets = generate_v7(
        sector_indices=sector_indices,
        sector_to_prices=sector_to_prices,
        describe=describe,
        cfg=cfg
    )
    print("=== sector_gate ===")
    print(sector_gate)
    # print("\n=== positions ===")
    # print(positions)
    out_path = r"CTA_" + asset_name + "_回看" + str(lookback) + "交易日.xlsx"
    save_v7_excel(out_path, positions, sector_gate, rank_sheets)
    # print("\nSaved:", out_path, "size:", os.path.getsize(out_path))
    pass

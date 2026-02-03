# Test_CTA_trend.py
# ------------------------------------------------------------
# Multi-asset (matrix) trend signals from daily close prices:
# (1) Vol-adjusted momentum:  R_k / sigma
# (2) t-stat momentum (basic): mean / (std/sqrt(n))
# (2b) t-stat momentum (Newey-West HAC): mean / se_HAC(mean)
# (3) z-score + tanh smoothing: tanh((x - mu)/sigma)
#
# Input: prices DataFrame (index=datetime, columns=assets), daily close
# Output: signals DataFrames aligned to prices index
# ------------------------------------------------------------

from __future__ import annotations

from Backtest import *

from numpy.lib.stride_tricks import sliding_window_view

import matplotlib

matplotlib.use('Agg')  # 设置不显示图片 必须在导入pyplot之前
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

'''##################################################################################################################'''


# =============================
# 构建因子
# =============================

def _to_returns(prices: pd.DataFrame, method: str = "pct") -> pd.DataFrame:
    """Daily returns from prices. method: 'pct' or 'log'."""
    if method == "log":
        return np.log(prices).diff()
    return prices.pct_change()


def clip_signal(x: pd.DataFrame | pd.Series, lo: float = -10.0, hi: float = 10.0):
    """Simple clipping to reduce extreme values (common in production)."""
    return x.clip(lower=lo, upper=hi)


# -----------------------------
# (1) Vol-adjusted momentum (matrix)
# Signal_t = R_{t-k,t} / sigma_{t-vol_lookback,t}
# -----------------------------
def compute_vol_adj_mom(
        prices: pd.DataFrame,
        lookback: int = 126,
        vol_lookback: int = 63,
        returns_method: str = "pct",
        annualize_vol: bool = True,
        clip: tuple[float, float] | None = (-10.0, 10.0),
) -> pd.DataFrame:
    """
    Vol-adjusted momentum per asset.
    - cum_ret: price/price.shift(k)-1
    - vol: rolling std of daily returns over vol_lookback (annualized by sqrt(252) if requested)
    """
    ret = _to_returns(prices, method=returns_method)
    cum_ret = prices / prices.shift(lookback) - 1.0

    vol = ret.rolling(vol_lookback).std()
    if annualize_vol:
        vol = vol * np.sqrt(252)

    sig = cum_ret / vol
    if clip is not None:
        sig = clip_signal(sig, clip[0], clip[1])
    return sig.dropna()


# -----------------------------
# (2) t-stat momentum (basic, matrix)
# t = mean(r) / (std(r)/sqrt(n))
# -----------------------------
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


# -----------------------------
# (2b) t-stat momentum with Newey-West HAC (matrix)
# Practical fast implementation for rolling windows:
# se_HAC(mean) = sqrt( (gamma0 + 2*sum_{l=1..L} w_l * gamma_l) / n )
# where gamma_l = autocov of demeaned returns at lag l,
# w_l = 1 - l/(L+1)  (Bartlett weights)
# t = mean / se_HAC(mean)
# -----------------------------
def compute_t_stat_nw(
        prices: pd.DataFrame,
        lookback: int = 126,
        lags: int = 5,
        returns_method: str = "pct",
        ddof: int = 0,
        clip: tuple[float, float] | None = (-10.0, 10.0),
) -> pd.DataFrame:
    """
    传统 t-stat 假设 iid，但金融序列存在：自相关、异方差。Newey-West 调整可以给出更稳健的标准误。
    思路：对每个滚动窗口，用 OLS 拟合截距模型，采用 Newey-West HAC 协方差矩阵估计截距参数的标准误，从而算 t-stat。
    （HAC：Heteroskedasticity and Autocorrelation Consistent，即异方差和自相关一致的）

    Rolling Newey-West t-stat for each asset (HAC standard error of the mean).
    Notes:
    - ddof controls variance/autocov normalization; ddof=0 is common for HAC math.（总体标准差）
    - This is time-series HAC; works for single asset or many assets.
    - Complexity: O(T * lags * N_assets) but vectorized over time inside each asset.
    """
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

        w = sliding_window_view(r, window_shape=lookback)  # (T-lookback+1, lookback)
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

        end_pos = np.arange(lookback - 1, len(r))  # window ends
        out.iloc[end_pos, out.columns.get_loc(col)] = t_full

    if clip is not None:
        out = clip_signal(out, clip[0], clip[1])
    return out.dropna()


# -----------------------------
# (3) z-score + tanh smoothing (matrix)
# Signal = tanh( (x - rolling_mean)/rolling_std )
# -----------------------------
def compute_tanh_zscore_old(
        x: pd.DataFrame,
        z_lookback: int = 252,
        min_periods: int | None = None,
        ddof: int = 0,
        clip_z: float | None = 8.0,
        tanh_k: float = 1.0,
) -> pd.DataFrame:
    if min_periods is None:
        min_periods = z_lookback

    mu = x.rolling(z_lookback, min_periods=min_periods).mean()
    sd = x.rolling(z_lookback, min_periods=min_periods).std(ddof=ddof)

    # 避免除以 0 / 极小波动导致爆炸
    sd = sd.replace(0.0, np.nan)

    z = (x - mu) / sd
    if clip_z is not None:
        z = z.clip(lower=-clip_z, upper=clip_z)

    out = np.tanh(tanh_k * z)
    # 显式保证返回 DataFrame（更稳）
    return pd.DataFrame(out, index=x.index, columns=x.columns)


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



def compute_tanh_signal(
    prices: pd.DataFrame,
    mom_lookback: int = 126,
    returns_method: str = "raw_mom",
    z_lookback: int = 252,
    min_periods: int | None = None,
    ddof: int = 0,
    clip_z: float | None = 8.0,
    tanh_k: float = 1.0,
) -> pd.DataFrame:
    """
    原始动量 -> rolling zscore -> tanh
    returns_method:
      - "raw_mom": prices/prices.shift(mom_lookback)-1 （v3 默认）
      - "pct_sum": lookback 内收益求和（可选）
      - "log_sum": lookback 内对数收益求和（可选）
    """
    if returns_method == "raw_mom":
        mom = prices / prices.shift(mom_lookback) - 1.0
    else:
        ret = prices.pct_change()
        if returns_method == "pct_sum":
            mom = ret.rolling(mom_lookback, min_periods=mom_lookback).sum()
        elif returns_method == "log_sum":
            logret = np.log1p(ret)
            mom = logret.rolling(mom_lookback, min_periods=mom_lookback).sum()
        else:
            raise ValueError(f"Unknown returns_method: {returns_method}")

    return compute_tanh_zscore(
        mom,
        z_lookback=z_lookback,
        min_periods=min_periods,
        ddof=ddof,
        clip_z=clip_z,
        tanh_k=tanh_k,
    )



'''##################################################################################################################'''


# =============================
# 构建权重
# =============================


def weight_longshort(
        factor_df: pd.DataFrame,
        hold_dates,
        top_n: int = 3,
        bottom_n: int = 3
):
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

        # 至少需要的资产数量
        min_required = max(top_n, 0) + max(bottom_n, 0)
        if len(x) < min_required or min_required == 0:
            continue

        x_sorted = x.sort_values(ascending=False)

        # ===== 多头 =====
        if top_n > 0:
            longs = x_sorted.head(top_n).index
            w_long = 1.0 / top_n
            for code in longs:
                rows.append({
                    "Date": d,
                    "S_INFO_WINDCODE": code,
                    "Weight": w_long
                })

        # ===== 空头 =====
        if bottom_n > 0:
            shorts = x_sorted.tail(bottom_n).index
            w_short = -1.0 / bottom_n
            for code in shorts:
                rows.append({
                    "Date": d,
                    "S_INFO_WINDCODE": code,
                    "Weight": w_short
                })

    Hold_df = pd.DataFrame(rows)
    if not Hold_df.empty:
        Hold_df = Hold_df.sort_values(
            ["Date", "S_INFO_WINDCODE"]
        ).reset_index(drop=True)

    return Hold_df


'''##################################################################################################################'''
# =============================
# 回测
# =============================


if __name__ == "__main__":
    asset_name = '黑色'
    position_frequency = 'week'

    price_table = pd.read_csv(
        r'D:\我的坚果云\1YF\my py\日参考模型\日频更新数据\日频趋势' + asset_name + '板块数据.csv', index_col=0,
        parse_dates=True)
    hold_dates = position_date(price_table, vPeriod=position_frequency)

    top_num = 1
    bottom_num = 0

    # '''(1)Vol-adjusted momentum'''
    # strategy_name = 'Vol-adjusted momentum'
    # for mon_lookback in [63, 126]:
    #     for vol_lookback in [26, 63]:
    #         sig1 = compute_vol_adj_mom(price_table, lookback=mon_lookback, vol_lookback=vol_lookback)
    #         Hold_df1 = weight_longshort(sig1, hold_dates, top_n=top_num, bottom_n=bottom_num)
    #         # Hold_df1.to_csv(
    #         #     asset_name + '_' + strategy_name + '策略_mon' + str(mon_lookback) + '天_vol' + str(
    #         #         vol_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
    #         #         bottom_num) + '空头_持仓.csv', index=False)
    #         # 回测净值
    #         nav1 = Back_test(Hold_df1, price_table, vfee=0.0003)
    #         # benchmark（买入持有板块指数）
    #         bench = (1 + price_table.iloc[:, 0].pct_change().fillna(0)).cumprod()
    #         bench = bench.to_frame("Nav")
    #         # 对齐日期（取交集）
    #         common_idx = bench.index.intersection(nav1.index).sort_values()
    #         nav1 = nav1.loc[common_idx].ffill()
    #         bench = bench.loc[common_idx].ffill()
    #         # 在“对齐后的起点”统一 rebasing
    #         nav1 = rebase_nav(nav1)
    #         bench = rebase_nav(bench)
    #         # 画图 & 保存
    #         plt.figure(figsize=(13, 7))
    #         plt.plot(bench.index, bench["Nav"].values, label=asset_name + '基准')
    #         plt.plot(nav1.index, nav1["Nav"].values, label=asset_name + '净值')
    #         plt.xlabel("Date")
    #         plt.ylabel("Net Asset Value")
    #         plt.title(asset_name + '_' + strategy_name + '策略_mon' + str(mon_lookback) + '天_vol' + str(
    #             vol_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
    #             bottom_num) + '空头_净值对比')
    #         plt.legend()
    #         plt.grid(True, alpha=0.3)
    #         plt.tight_layout()
    #         out_png = asset_name + '_' + strategy_name + '策略_mon' + str(mon_lookback) + '天_vol' + str(
    #             vol_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
    #             bottom_num) + '空头_净值对比.png'
    #         plt.savefig(out_png, dpi=200)
    #         plt.close()

    '''(2)t-stat momentum'''
    strategy_name = 't-stat momentum'
    for mon_lookback in [5]:
        sig2 = compute_t_stat_nw(price_table, lookback=mon_lookback, lags=5)
        Hold_df2 = weight_longshort(sig2, hold_dates, top_n=top_num, bottom_n=bottom_num)
        # Hold_df2.to_csv(
        #     asset_name + '_' + strategy_name + '策略_mon' + str(
        #         mon_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
        #         bottom_num) + '空头_持仓.csv', index=False)
        # 回测净值
        nav2 = Back_test(Hold_df2, price_table, vfee=0.0003)
        # benchmark（买入持有板块指数）
        bench = (1 + price_table.iloc[:, 0].pct_change().fillna(0)).cumprod()
        bench = bench.to_frame("Nav")
        # 对齐日期（取交集）
        common_idx = bench.index.intersection(nav2.index).sort_values()
        nav2 = nav2.loc[common_idx].ffill()
        bench = bench.loc[common_idx].ffill()
        # 在“对齐后的起点”统一 rebasing
        nav2 = rebase_nav(nav2)
        bench = rebase_nav(bench)
        # 画图 & 保存
        plt.figure(figsize=(13, 7))
        plt.plot(bench.index, bench["Nav"].values, label=asset_name + '基准')
        plt.plot(nav2.index, nav2["Nav"].values, label=asset_name + '净值')
        plt.xlabel("Date")
        plt.ylabel("Net Asset Value")
        plt.title(asset_name + '_' + strategy_name + '策略_mon' + str(
            mon_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
            bottom_num) + '空头_净值对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = asset_name + '_' + strategy_name + '策略_mon' + str(
            mon_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
            bottom_num) + '空头_净值对比.png'
        plt.savefig(out_png, dpi=200)
        plt.close()

    # '''(3)z-score & tanh smoothing'''
    # strategy_name = 'z-score & tanh smoothing'
    # for mon_lookback in [63, 126, 252]:
    #     for z_lookback in [126, 252]:
    #         sig3_ = compute_tanh_signal(prices=price_table, mom_lookback=mon_lookback, z_lookback=z_lookback)
    #         Hold_df3_ = weight_longshort(sig3_, hold_dates, top_n=top_num, bottom_n=bottom_num)
    #         # Hold_df3_.to_csv(
    #         #     asset_name + '_' + strategy_name + '策略_mon' + str(
    #         #         mon_lookback) + '天_z' + str(
    #         #         z_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
    #         #         bottom_num) + '空头_持仓.csv', index=False)
    #         # 回测净值
    #         nav3_ = Back_test(Hold_df3_, price_table, vfee=0.0003)
    #         # benchmark（买入持有板块指数）
    #         bench = (1 + price_table.iloc[:, 0].pct_change().fillna(0)).cumprod()
    #         bench = bench.to_frame("Nav")
    #         # 对齐日期（取交集）
    #         common_idx = bench.index.intersection(nav3_.index).sort_values()
    #         nav3_ = nav3_.loc[common_idx].ffill()
    #         bench = bench.loc[common_idx].ffill()
    #         # 在“对齐后的起点”统一 rebasing
    #         nav3_ = rebase_nav(nav3_)
    #         bench = rebase_nav(bench)
    #         # 画图 & 保存
    #         plt.figure(figsize=(13, 7))
    #         plt.plot(bench.index, bench["Nav"].values, label=asset_name + '基准')
    #         plt.plot(nav3_.index, nav3_["Nav"].values, label=asset_name + '净值')
    #         plt.xlabel("Date")
    #         plt.ylabel("Net Asset Value")
    #         plt.title(asset_name + '_' + strategy_name + '策略_mon' + str(
    #             mon_lookback) + '天_z' + str(
    #             z_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
    #             bottom_num) + '空头_净值对比')
    #         plt.legend()
    #         plt.grid(True, alpha=0.3)
    #         plt.tight_layout()
    #         out_png = asset_name + '_' + strategy_name + '策略_mon' + str(
    #             mon_lookback) + '天_z' + str(
    #             z_lookback) + '天_' + position_frequency + '换仓_' + str(top_num) + '多头_' + str(
    #             bottom_num) + '空头_净值对比.png'
    #         plt.savefig(out_png, dpi=200)
    #         plt.close()

'''
(1) vol-adjusted momentum
prices  →  raw_mom

(2) t-stat momentum
prices  →  t_stat_mom

(3) z-score + tanh
raw_mom / t_stat_mom / any_signal
        →  tanh_zscore_signal
'''

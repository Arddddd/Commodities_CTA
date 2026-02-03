import pandas as pd
import datetime
import numpy as np


# 相关系数矩阵变为协方差矩阵
def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


# 协方差矩阵变为相关系数矩阵
def cov2corr(cov):
    std = np.sqrt(np.diag(cov))  # cov_df对角开平方根得出一个向量
    corr = cov / np.outer(std, std)  # R = C / sqrt(diag(C)) ，C满足多元正态分布
    # np.outer(a,b)计算矩阵的外积，把a当做列向量，b当做行向量，得到一个n*m的矩阵
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


# 获得调仓日期数据
def position_date_old(all_data, vPeriod="month", Is_yc="ym"):
    '''
    vPeriod：week周频/halfmonth半月频/month/quarter/halfyear/year
    Is_yc：yc期初/ym期末
    '''
    if vPeriod == "week" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = [Date_arr[0]]
        s_date = Date_arr[0]
        Hold_date.append(s_date)
        for i in range(len(Date_arr) - 2):
            if (pd.to_datetime(Date_arr[i]) - pd.to_datetime(s_date)).days >= 7:
                Hold_date.append(Date_arr[i + 1])
                s_date = Date_arr[i]
    # 半月频（两周换一次仓）：期初调仓
    # 规则与周频期初一致：当距离上次调仓日 >= 14 天，则在“下一交易日”调仓
    # 说明：这里的 vPeriod 取值建议使用 "halfmonth"
    if vPeriod == "halfmonth" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = [Date_arr[0]]
        s_date = Date_arr[0]
        for i in range(len(Date_arr) - 2):
            if (pd.to_datetime(Date_arr[i]) - pd.to_datetime(s_date)).days >= 14:
                Hold_date.append(Date_arr[i + 1])
                s_date = Date_arr[i]
    if vPeriod == "month" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = [Date_arr[0]]
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                Hold_date.append(Date_arr[i + 1])
    if vPeriod == "quarter" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = [Date_arr[0]]
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i + 1])
        Hold_date = [Date_arr[0]]
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 65:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    if vPeriod == "halfyear" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = [Date_arr[0]]
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i + 1])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 175:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    if vPeriod == "year" and Is_yc == 'yc':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = [Date_arr[0]]
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i + 1])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 350:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    # 获得调仓日期数据
    if vPeriod == "week" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = []
        s_date = Date_arr[0]
        Hold_date.append(s_date)
        for i in range(len(Date_arr) - 2):
            if (pd.to_datetime(Date_arr[i]) - pd.to_datetime(s_date)).days >= 7:
                Hold_date.append(Date_arr[i])
                s_date = Date_arr[i]
    # 半月频（两周换一次仓）：期末调仓
    # 规则与周频期末一致：当距离上次调仓日 >= 14 天，则用“当前交易日”作为调仓日
    if vPeriod == "halfmonth" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = []
        s_date = Date_arr[0]
        Hold_date.append(s_date)
        for i in range(len(Date_arr) - 2):
            if (pd.to_datetime(Date_arr[i]) - pd.to_datetime(s_date)).days >= 14:
                Hold_date.append(Date_arr[i])
                s_date = Date_arr[i]
    if vPeriod == "month" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        Hold_date = []
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):  # 截出月份，若月份不相同则已换月
                Hold_date.append(Date_arr[i])  # 筛出月底日期
    if vPeriod == "quarter" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = []
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 65:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    if vPeriod == "halfyear" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = []
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 175:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    if vPeriod == "year" and Is_yc == 'ym':
        Date_arr = all_data.index.astype(str).unique()
        tmp_Hold_date = []
        for i in range(len(Date_arr) - 2):
            if (Date_arr[i][0:7] != Date_arr[i + 1][0:7]):
                tmp_Hold_date.append(Date_arr[i])
        Hold_date = []
        s_date = tmp_Hold_date[0]
        Hold_date.append(s_date)
        for i in range(len(tmp_Hold_date) - 2):
            if (pd.to_datetime(tmp_Hold_date[i]) - pd.to_datetime(s_date)).days >= 350:
                Hold_date.append(tmp_Hold_date[i])
                s_date = tmp_Hold_date[i]
    return Hold_date


# 获得调仓权重数据
def position_date(all_data, vPeriod="week"):
    """
    每周五形成一个仓位，下周一用该仓位计算净值。
    （假设本周到下周有假期，那就用本周最后一个工作日的日期形成仓位，下周第一个工作日用该仓位计算净值）
    vPeriod：week / halfmonth / month / halfyear / year
    返回：形成仓位日（周期末最后一个交易日）。在 Back_test 中会从下一交易日开始生效。
    """
    idx = pd.to_datetime(all_data.index)
    idx = idx.sort_values()

    if len(idx) == 0:
        return []

    # --- 1) 周频：每周最后一个交易日 ---
    if vPeriod == "week":
        # to_period('W') 会按自然周分组（周一~周日），我们取每组最大日期 = 该周最后一个交易日
        hold = idx.to_series().groupby(idx.to_period("W")).max().tolist()
        return [d.strftime("%Y-%m-%d") for d in hold]

    # --- 2) 半月频（两周）：取“周末最后一个交易日”的序列，再每隔一周取一次 ---
    if vPeriod == "halfmonth":
        week_ends = idx.to_series().groupby(idx.to_period("W")).max().tolist()
        hold = week_ends[::2]  # 每两周一次（第1、3、5...个周末）
        return [d.strftime("%Y-%m-%d") for d in hold]

    # --- 3) 月频：每月最后一个交易日 ---
    if vPeriod == "month":
        hold = idx.to_series().groupby(idx.to_period("M")).max().tolist()
        return [d.strftime("%Y-%m-%d") for d in hold]

    # --- 4) 半年频：每半年最后一个交易日（上半年/下半年）---
    if vPeriod == "halfyear":
        # key: (year, half) half=1 for Jan-Jun, 2 for Jul-Dec
        half_key = pd.Index([(d.year, 1 if d.month <= 6 else 2) for d in idx])
        hold = idx.to_series().groupby(half_key).max().tolist()
        return [d.strftime("%Y-%m-%d") for d in hold]

    # --- 5) 年频：每年最后一个交易日 ---
    if vPeriod == "year":
        hold = idx.to_series().groupby(idx.to_period("Y")).max().tolist()
        return [d.strftime("%Y-%m-%d") for d in hold]

    raise ValueError(f"Unsupported vPeriod: {vPeriod}")


# 回测函数
def Back_test_old(Hold_df, price_table, vfee=0.0003):
    '''
    Hold_df:权重
    price_table:价格（一般为收盘价）
    '''

    # 初始净值为1
    start_nav = 1

    vData = Hold_df.copy()
    # 权重df的日期转成 datetime 并排序
    vData['Date'] = pd.to_datetime(vData['Date'])
    Date_arr = np.sort(vData['Date'].unique())

    # 价格df的日期也统一成 datetime
    price_table.index = pd.to_datetime(price_table.index)
    price_table = price_table.sort_index()

    nav = pd.DataFrame()

    for i_date in Date_arr:
        tmp_vData = vData[vData['Date'] == i_date]  # i_date 的权重
        tmp_date = Date_arr[Date_arr > i_date]  # i_date之后的交易日（在所有调仓日 Date_arr里，找出比 i_date 更晚的那些调仓日）
        if len(tmp_date) > 0:
            end_date = tmp_date[0]  # 如果还有下一次仓位日，那么 end_date = 下一次仓位日
        else:
            end_date = price_table.index.max()  # 如果 i_date 是最后一个仓位日，那就假设这套仓位一直持有到现在

        # i_date后到end_date（含）期间的价格序列
        tmp_price = price_table[(price_table.index > i_date) & (price_table.index <= end_date)][
            tmp_vData['S_INFO_WINDCODE']
        ]
        if tmp_price.empty:
            continue

        tmp_return = tmp_price.pct_change(1).fillna(0)

        # ---- L仅作为监控指标：gross exposure（不参与计算） ----
        L = np.sum(np.abs(tmp_vData['Weight'].values))
        if L == 0:
            tmp_nav = np.ones(len(tmp_return)) * start_nav
            tmp_nav_df = pd.DataFrame(tmp_nav, index=tmp_price.index, columns=['Nav'])
            nav = pd.concat([nav, tmp_nav_df])
            start_nav = tmp_nav[-1]
            continue

        # ---- 核心改动：用组合日收益计算净值（兼容负权重、sum(w)任意）----
        w = (
            tmp_vData.set_index('S_INFO_WINDCODE')['Weight']
            .reindex(tmp_return.columns)
            .fillna(0.0)
            .values
        )
        port_ret = tmp_return.values @ w  # r_p,t = Σ w_i r_i,t
        tmp_nav = np.cumprod(1 + port_ret) * start_nav * (1 - vfee)

        start_nav = tmp_nav[-1]
        tmp_nav_df = pd.DataFrame(tmp_nav, index=tmp_price.index, columns=['Nav'])
        nav = pd.concat([nav, tmp_nav_df])

    return nav


def Back_test(Hold_df, price_table, vfee=0.0003):
    start_nav = 1.0

    vData = Hold_df.copy()
    vData["Date"] = pd.to_datetime(vData["Date"])

    price_table = price_table.copy()
    price_table.index = pd.to_datetime(price_table.index)
    price_table = price_table.sort_index()

    # 关键：调仓日排序
    Date_arr = np.sort(vData["Date"].unique())

    nav_list = []

    for idx_d, i_date in enumerate(Date_arr):
        tmp_vData = vData[vData["Date"] == i_date]

        # end_date：下一次“形成权重日”（不是生效日）
        if idx_d < len(Date_arr) - 1:
            end_date = Date_arr[idx_d + 1]
        else:
            end_date = price_table.index.max()

        # 区间：(i_date, end_date] —— i_date 次日生效；end_date 当天仍用旧权重
        cols = tmp_vData["S_INFO_WINDCODE"].unique()
        cols = [c for c in cols if c in price_table.columns]

        # 若权重资产在价格表中一个都找不到，则视为全现金：净值不变覆盖该区间
        tmp_price = price_table.loc[(price_table.index > i_date) & (price_table.index <= end_date), cols]
        if tmp_price.empty:
            continue

        tmp_return = tmp_price.pct_change().fillna(0.0)

        # gross exposure 监控（不参与计算）
        L = float(np.sum(np.abs(tmp_vData["Weight"].values)))

        if L == 0:
            tmp_nav = np.full(len(tmp_return), start_nav, dtype=float)
        else:
            w = (
                tmp_vData.set_index("S_INFO_WINDCODE")["Weight"]
                .reindex(tmp_return.columns)
                .fillna(0.0)
                .values.astype(float)
            )
            port_ret = tmp_return.values @ w
            tmp_nav = np.cumprod(1.0 + port_ret) * start_nav

            # 固定费用：每段扣一次（段首扣）
            tmp_nav = tmp_nav * (1.0 - vfee)

        start_nav = float(tmp_nav[-1])
        nav_list.append(pd.DataFrame({"Nav": tmp_nav}, index=tmp_return.index))

    if len(nav_list) == 0:
        return pd.DataFrame(columns=["Nav"])

    nav = pd.concat(nav_list).sort_index()
    # 万一有重复日期（例如数据异常），保留最后一个
    nav = nav[~nav.index.duplicated(keep="last")]

    return nav


# 将净值改为从1开始的序列
def rebase_nav(nav: pd.DataFrame, base=1.0):
    return nav / nav.iloc[0] * base


# 计算累计收益函数
def accumReturn(Asset_nav):
    Asset_nav_values = Asset_nav.values
    accum_return = Asset_nav_values[len(Asset_nav_values) - 1] / Asset_nav_values[0] - 1
    return round(accum_return, 3)


# 计算年化收益函数
def annReturn(Asset_nav, annual_day=240):
    Asset_nav_values = Asset_nav.values
    during_day = len(Asset_nav_values)
    annual_rate = (Asset_nav_values[len(Asset_nav_values) - 1] / Asset_nav_values[0]) ** (annual_day / during_day) - 1
    return round(annual_rate, 3)


# 计算年化波动率
def annVolatility(Asset_nav, annual_day=240):
    ret = Asset_nav.pct_change(1).dropna()
    ann_vol = np.std(ret) * np.sqrt(annual_day)
    return round(ann_vol, 3)


# 计算滚动年化波动率（时间序列）
def RollannVolatility(Asset_nav, annual_day=240, windows=60):
    ret = Asset_nav.pct_change(1).dropna()
    roll_ann_vol = (ret.rolling(windows).std()) * np.sqrt(annual_day)
    roll_ann_vol = roll_ann_vol.dropna()
    return roll_ann_vol


# 计算年化夏普
def sharpRatio(Asset_nav, annual_day=240):
    sharp_ratio = annReturn(Asset_nav, annual_day) / annVolatility(Asset_nav, annual_day)
    return round(sharp_ratio, 3)


# 计算最大回撤
def max_drawdown(Asset_nav):
    Asset_nav_values = Asset_nav.values
    acc_max = np.maximum.accumulate(Asset_nav_values)
    max_drawdown = np.max((acc_max - Asset_nav_values) / acc_max)
    return round(max_drawdown, 3)

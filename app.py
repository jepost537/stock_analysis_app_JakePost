"""
Stock Comparison & Analysis Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from scipy import stats
from datetime import date, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
)

TRADING_DAYS = 252  # annualization constant throughout

# ═════════════════════════════════════════════════════════════════════════════
# CACHED DATA FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def download_data(tickers: tuple, start: str, end: str):
    """
    Download adjusted closing prices for user tickers + S&P 500.
    Returns (df_prices | None, list_of_warning_strings).
    """
    all_tickers = list(tickers) + ["^GSPC"]
    errors = []

    try:
        raw = yf.download(
            all_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        return None, [f"Download failed: {e}"]

    if raw.empty:
        return None, ["Yahoo Finance returned no data. Check your tickers and date range."]

    # yfinance returns MultiIndex columns for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"]
    else:
        # Single ticker edge case — wrap in DataFrame
        df = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    if isinstance(df, pd.Series):
        df = df.to_frame(name=all_tickers[0])

    # Flag tickers that came back empty
    for t in all_tickers:
        if t not in df.columns:
            errors.append(f"'{t}': not found in downloaded data (invalid ticker?).")
        elif df[t].isna().all():
            errors.append(f"'{t}': all values are NaN — likely an invalid ticker.")

    # Drop fully-empty columns
    df = df.dropna(axis=1, how="all")

    if df.empty:
        return None, errors + ["No valid price data returned."]

    # Drop tickers with >5% missing values and warn
    to_drop = []
    for col in df.columns:
        pct_missing = df[col].isna().mean()
        if pct_missing > 0.05:
            to_drop.append(col)
            errors.append(
                f"'{col}' dropped — {pct_missing:.1%} of values are missing (>5% threshold)."
            )
    if to_drop:
        df = df.drop(columns=to_drop)

    # Truncate to overlapping date range (drop any row with a NaN in any column)
    pre_rows = len(df)
    df = df.dropna(how="any")
    if len(df) < pre_rows:
        errors.append(
            f"Date range truncated to overlapping data: "
            f"{df.index.min().date()} to {df.index.max().date()}."
        )

    if df.empty:
        return None, errors + ["No overlapping data after removing missing values."]

    return df, errors


@st.cache_data
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily simple (arithmetic) returns: pct_change().dropna()"""
    return df.pct_change().dropna()


@st.cache_data
def compute_summary_stats(returns: pd.DataFrame, tickers: tuple) -> pd.DataFrame:
    """
    Summary statistics table.
    Annualization: mean * 252  |  vol * sqrt(252)  — per project spec section 7.1
    """
    cols = list(tickers)
    r = returns[cols]
    return pd.DataFrame(
        {
            "Ann. Mean Return": r.mean() * TRADING_DAYS,
            "Ann. Volatility":  r.std()  * np.sqrt(TRADING_DAYS),
            "Skewness":         r.skew(),
            "Excess Kurtosis":  r.kurt(),
            "Min Daily Return": r.min(),
            "Max Daily Return": r.max(),
        }
    )


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUTS + ABOUT
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📈 Stock Analyzer")
    st.markdown("---")

    ticker_input = st.text_input(
        "Enter 2–5 stock tickers (comma-separated)",
        value="AAPL, MSFT, NVDA",
        help="Examples: AAPL, MSFT, GOOGL, AMZN, META",
    )

    default_end   = date.today()
    default_start = default_end - timedelta(days=3 * 365)

    start_date = st.date_input("Start Date", value=default_start)
    end_date   = st.date_input("End Date",   value=default_end)

    load_btn = st.button("🚀 Load Data", type="primary", use_container_width=True)

    st.markdown("---")

    with st.expander("ℹ️ About & Methodology"):
        st.markdown(
            """
**What this app does**
Compare and analyze multiple stocks using historical adjusted close price data from Yahoo Finance.

**Key assumptions**
- Returns are **simple (arithmetic)**: `pct_change()`
- Annualization uses **252 trading days** per year
- Annualized return = mean daily return × 252
- Annualized volatility = daily std × √252
- Cumulative returns: `(1 + r).cumprod()`
- Equal-weight portfolio return = average of daily returns across all selected stocks

**Data source**
Yahoo Finance via `yfinance` (adjusted close prices).
`^GSPC` (S&P 500) is used as a benchmark only — not a portfolio constituent.
            """
        )

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("df_prices",     None),
    ("daily_returns", None),
    ("user_tickers",  []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

if load_btn:
    raw_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    raw_tickers = list(dict.fromkeys(raw_tickers))  # deduplicate, preserve order

    if len(raw_tickers) < 2:
        st.sidebar.error("Please enter at least 2 ticker symbols.")
    elif len(raw_tickers) > 5:
        st.sidebar.error("Please enter no more than 5 ticker symbols.")
    elif end_date <= start_date:
        st.sidebar.error("End date must be after start date.")
    elif (end_date - start_date).days < 365:
        st.sidebar.error("Date range must be at least 1 year.")
    else:
        with st.spinner("Downloading data from Yahoo Finance…"):
            df, errors = download_data(
                tuple(raw_tickers), str(start_date), str(end_date)
            )

        for msg in errors:
            st.sidebar.warning(msg)

        if df is None or df.empty:
            st.sidebar.error("No data loaded. Please check your tickers and try again.")
        else:
            user_tickers = [t for t in raw_tickers if t in df.columns]

            if len(user_tickers) < 2:
                st.sidebar.error(
                    "Fewer than 2 valid tickers remain after data validation. "
                    "Please adjust your inputs."
                )
            else:
                st.session_state.df_prices    = df
                st.session_state.daily_returns = compute_returns(df)
                st.session_state.user_tickers  = user_tickers
                st.sidebar.success(
                    f"Loaded: {', '.join(user_tickers)}"
                    + (" + S&P 500 benchmark" if "^GSPC" in df.columns else "")
                )

# ═════════════════════════════════════════════════════════════════════════════
# GUARD — nothing loaded yet
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.df_prices is None:
    st.title("📈 Stock Comparison & Analysis App")
    st.info(
        "👈 Enter 2–5 ticker symbols and a date range in the sidebar, "
        "then click **🚀 Load Data** to begin."
    )
    st.stop()

# Convenience aliases
df           = st.session_state.df_prices
returns      = st.session_state.daily_returns
user_tickers = st.session_state.user_tickers
has_sp       = "^GSPC" in df.columns

# ═════════════════════════════════════════════════════════════════════════════
# THREE MAIN TABS
# ═════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊 Price & Returns",
    "📉 Risk & Distribution",
    "🔗 Correlation & Diversification",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PRICE & RETURN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Price & Return Analysis")

    # ── 1a. Adjusted closing price chart ─────────────────────────────────────
    st.subheader("Adjusted Closing Prices")

    price_select = st.multiselect(
        "Select stocks to display",
        options=user_tickers,
        default=user_tickers,
        key="price_select",
    )

    if price_select:
        fig_price = go.Figure()
        for t in price_select:
            fig_price.add_trace(
                go.Scatter(x=df.index, y=df[t], name=t, mode="lines")
            )
        fig_price.update_layout(
            title="Adjusted Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            height=440,
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.warning("Select at least one stock to display the price chart.")

    # ── 1b. Summary statistics ────────────────────────────────────────────────
    st.subheader("Summary Statistics")

    stat_tickers = tuple(user_tickers + (["^GSPC"] if has_sp else []))
    stats_df = compute_summary_stats(returns, stat_tickers)

    display_stats = stats_df.copy()
    for col in ["Ann. Mean Return", "Ann. Volatility", "Min Daily Return", "Max Daily Return"]:
        display_stats[col] = display_stats[col].map("{:.2%}".format)
    display_stats["Skewness"]       = display_stats["Skewness"].map("{:.4f}".format)
    display_stats["Excess Kurtosis"] = display_stats["Excess Kurtosis"].map("{:.4f}".format)

    st.dataframe(display_stats, use_container_width=True)
    st.caption(
        "Annualized return = mean daily return × 252. "
        "Annualized volatility = daily std × √252. "
        "Excess kurtosis: normal distribution = 0."
    )

    # ── 1c. Cumulative wealth index ───────────────────────────────────────────
    with st.expander("View Daily Returns Data"):
        st.dataframe(returns[user_tickers].style.format("{:.4%}"), use_container_width=True)
    st.subheader("Cumulative Wealth Index — $10,000 Invested")

    wealth    = (1 + returns[user_tickers]).cumprod() * 10_000
    ew_daily  = returns[user_tickers].mean(axis=1)        # equal-weight daily return
    ew_wealth = (1 + ew_daily).cumprod() * 10_000

    fig_wealth = go.Figure()
    for t in user_tickers:
        fig_wealth.add_trace(
            go.Scatter(x=wealth.index, y=wealth[t], name=t, mode="lines")
        )

    if has_sp:
        sp_wealth = (1 + returns["^GSPC"]).cumprod() * 10_000
        fig_wealth.add_trace(
            go.Scatter(
                x=sp_wealth.index, y=sp_wealth,
                name="S&P 500", mode="lines",
                line=dict(color="black", dash="dash", width=2),
            )
        )

    fig_wealth.add_trace(
        go.Scatter(
            x=ew_wealth.index, y=ew_wealth,
            name="Equal-Weight Portfolio", mode="lines",
            line=dict(color="gold", dash="dot", width=2),
        )
    )

    fig_wealth.update_layout(
        title="Growth of $10,000",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig_wealth, use_container_width=True)
    st.caption(
        "Wealth index = (1 + r).cumprod() × $10,000. "
        "Equal-Weight Portfolio: each day's return = simple average of all selected stocks."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RISK & DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Risk & Distribution Analysis")

    # ── 2a. Rolling annualized volatility ─────────────────────────────────────
    st.subheader("Rolling Annualized Volatility")

    roll_window = st.select_slider(
        "Rolling window (trading days)",
        options=[30, 60, 90, 126, 252],
        value=60,
        key="roll_window",
    )

    rolling_vol = (
        returns[user_tickers].rolling(roll_window).std() * np.sqrt(TRADING_DAYS)
    )

    fig_rvol = go.Figure()
    for t in user_tickers:
        fig_rvol.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol[t], name=t, mode="lines")
        )
    if has_sp:
        bm_vol = returns["^GSPC"].rolling(roll_window).std() * np.sqrt(TRADING_DAYS)
        fig_rvol.add_trace(
            go.Scatter(
                x=bm_vol.index, y=bm_vol,
                name="S&P 500", mode="lines",
                line=dict(color="black", dash="dash", width=2),
            )
        )

    fig_rvol.update_layout(
        title=f"{roll_window}-Day Rolling Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_rvol, use_container_width=True)
    st.caption(
        f"Rolling std dev × √252. "
        f"The first {roll_window - 1} observations are NaN — this is expected."
    )

    # ── 2b. Distribution analysis (histogram / Q-Q toggle) ────────────────────
    st.subheader("Return Distribution Analysis")

    dist_ticker = st.selectbox(
        "Select stock for distribution analysis",
        user_tickers,
        key="dist_ticker",
    )
    ret_series = returns[dist_ticker].dropna()

    # Jarque-Bera normality test
    jb_stat, jb_pval = stats.jarque_bera(ret_series)
    verdict = (
        "❌ Rejects normality (p < 0.05)"
        if jb_pval < 0.05
        else "✅ Fails to reject normality (p ≥ 0.05)"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Jarque-Bera Statistic", f"{jb_stat:,.2f}")
    c2.metric("p-value", f"{jb_pval:.2e}")
    c3.metric("Normality Test (5% level)", verdict)

    dist_view = st.radio(
        "Plot type",
        ["Histogram + Normal Fit", "Q-Q Plot"],
        horizontal=True,
        key="dist_view",
    )

    if dist_view == "Histogram + Normal Fit":
        mu_fit, sigma_fit = stats.norm.fit(ret_series)
        x_vals  = np.linspace(ret_series.min(), ret_series.max(), 400)
        pdf_vals = stats.norm.pdf(x_vals, loc=mu_fit, scale=sigma_fit)

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=ret_series,
                nbinsx=150,
                histnorm="probability density",
                name="Observed Returns",
                marker_color="steelblue",
                opacity=0.65,
            )
        )
        fig_hist.add_trace(
            go.Scatter(
                x=x_vals,
                y=pdf_vals,
                name=f"Normal Fit  (μ={mu_fit:.5f}, σ={sigma_fit:.5f})",
                mode="lines",
                line=dict(color="red", width=2),
            )
        )
        fig_hist.update_layout(
            title=f"Daily Return Distribution — {dist_ticker}",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            barmode="overlay",
            height=430,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            "Normal distribution fitted using scipy.stats.norm.fit() (maximum likelihood)."
        )

    else:  # Q-Q Plot
        (theoretical_q, ordered_vals), (slope, intercept, _) = stats.probplot(
            ret_series, dist="norm"
        )
        qq_line = slope * np.array(theoretical_q) + intercept

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_q,
                y=ordered_vals,
                mode="markers",
                name="Observed",
                marker=dict(color="steelblue", size=3, opacity=0.45),
            )
        )
        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_q,
                y=qq_line,
                mode="lines",
                name="Normal Reference Line",
                line=dict(color="red", width=2),
            )
        )
        fig_qq.update_layout(
            title=f"Normal Q-Q Plot — {dist_ticker}",
            xaxis_title="Theoretical Quantiles (Normal)",
            yaxis_title="Observed Quantiles",
            height=450,
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption(
            "Points on the red line → consistent with normality. "
            "S-shaped deviations at the tails → fat tails (leptokurtosis), "
            "which is typical of daily stock returns."
        )

    # ── 2c. Box plots ─────────────────────────────────────────────────────────
    st.subheader("Daily Return Distributions — Box Plots")

    fig_box = go.Figure()
    for t in user_tickers:
        fig_box.add_trace(
            go.Box(y=returns[t], name=t, boxmean=True, marker_size=2)
        )

    fig_box.update_layout(
        title="Box Plots of Daily Returns (all selected stocks)",
        xaxis_title="Ticker",
        yaxis_title="Daily Return",
        yaxis_tickformat=".1%",
        showlegend=False,
        height=430,
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption(
        "Line inside box = median. Cross = mean. "
        "Points beyond whiskers are outliers."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CORRELATION & DIVERSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Correlation & Diversification")

    # ── 3a. Correlation heatmap ───────────────────────────────────────────────
    st.subheader("Correlation Heatmap")

    corr         = returns[user_tickers].corr()
    corr_rounded = corr.round(2)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=corr_rounded.values,
            x=corr_rounded.columns.tolist(),
            y=corr_rounded.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_rounded.values,
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        )
    )
    fig_heat.update_layout(
        title="Pairwise Pearson Correlation — Daily Returns",
        height=max(350, 90 * len(user_tickers) + 100),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Diverging color scale centered at zero: red = positive, blue = negative.")

    # ── 3b. Scatter plot ──────────────────────────────────────────────────────
    st.subheader("Return Scatter Plot")

    default_b_idx = min(1, len(user_tickers) - 1)
    sc1, sc2 = st.columns(2)
    sc_a = sc1.selectbox("Stock A", user_tickers, index=0,              key="sc_a")
    sc_b = sc2.selectbox("Stock B", user_tickers, index=default_b_idx,  key="sc_b")

    if sc_a == sc_b:
        st.warning("Please select two different stocks for the scatter plot.")
    else:
        corr_val = returns[sc_a].corr(returns[sc_b])
        fig_sc = go.Figure(
            data=go.Scatter(
                x=returns[sc_a],
                y=returns[sc_b],
                mode="markers",
                marker=dict(size=4, opacity=0.35, color="steelblue"),
                text=returns.index.strftime("%Y-%m-%d"),
                hovertemplate="%{text}<br>%{x:.4f} / %{y:.4f}",
            )
        )
        fig_sc.update_layout(
            title=f"Daily Returns: {sc_a} vs {sc_b}   (ρ = {corr_val:.3f})",
            xaxis_title=f"{sc_a} Daily Return",
            yaxis_title=f"{sc_b} Daily Return",
            xaxis_tickformat=".1%",
            yaxis_tickformat=".1%",
            height=450,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── 3c. Rolling correlation ───────────────────────────────────────────────
    st.subheader("Rolling Correlation")

    rc1, rc2, rc3 = st.columns(3)
    rc_a   = rc1.selectbox("Stock A", user_tickers, index=0,             key="rc_a")
    rc_b   = rc2.selectbox("Stock B", user_tickers, index=default_b_idx, key="rc_b")
    rc_win = rc3.select_slider(
        "Window (days)", options=[30, 60, 90, 126, 252], value=60, key="rc_win"
    )

    if rc_a == rc_b:
        st.warning("Please select two different stocks for the rolling correlation.")
    else:
        roll_corr = returns[rc_a].rolling(rc_win).corr(returns[rc_b])
        full_corr = returns[rc_a].corr(returns[rc_b])

        fig_rc = go.Figure()
        fig_rc.add_trace(
            go.Scatter(
                x=roll_corr.index, y=roll_corr,
                name=f"{rc_win}-day rolling",
                mode="lines",
                line=dict(color="steelblue"),
            )
        )
        fig_rc.add_hline(
            y=full_corr,
            line_dash="dash",
            line_color="grey",
            annotation_text=f"Full-sample: {full_corr:.3f}",
        )
        fig_rc.update_layout(
            title=f"Rolling Correlation: {rc_a} & {rc_b}",
            xaxis_title="Date",
            yaxis_title="Correlation",
            yaxis_range=[-1, 1],
            height=390,
        )
        st.plotly_chart(fig_rc, use_container_width=True)

    # ── 3d. Two-asset portfolio explorer ─────────────────────────────────────
    st.subheader("Two-Asset Portfolio Explorer")

    st.info(
        "**Diversification in action:** Combining two stocks can produce a portfolio with "
        "*lower* volatility than either stock individually. The curve below dips beneath "
        "both endpoints when the correlation between the two stocks is less than 1. "
        "The lower the correlation, the deeper the dip — and the greater the diversification benefit."
    )

    pe1, pe2 = st.columns(2)
    port_a = pe1.selectbox("Stock A", user_tickers, index=0,             key="port_a")
    port_b = pe2.selectbox("Stock B", user_tickers, index=default_b_idx, key="port_b")

    if port_a == port_b:
        st.warning("Please select two different stocks for the portfolio explorer.")
    else:
        w_a_pct = st.slider(
            f"Weight in {port_a} (%)",
            min_value=0, max_value=100, value=50, step=1,
            key="w_a",
        )
        w_b_pct = 100 - w_a_pct
        st.caption(f"Weight in {port_b}: **{w_b_pct}%**")

        # Annualized building blocks — spec §7.1: mean*252, std*sqrt(252)
        ann_mu_a  = returns[port_a].mean() * TRADING_DAYS
        ann_mu_b  = returns[port_b].mean() * TRADING_DAYS
        ann_vol_a = returns[port_a].std()  * np.sqrt(TRADING_DAYS)
        ann_vol_b = returns[port_b].std()  * np.sqrt(TRADING_DAYS)
        # Annualized covariance: daily cov × 252  (since ann_var = daily_var × 252)
        ann_cov   = returns[port_a].cov(returns[port_b]) * TRADING_DAYS

        w = w_a_pct / 100

        # Two-asset portfolio formulas from spec §7.1
        port_ret = w * ann_mu_a + (1 - w) * ann_mu_b
        port_var = (
            w**2       * ann_vol_a**2
            + (1-w)**2 * ann_vol_b**2
            + 2 * w * (1-w) * ann_cov
        )
        port_vol = np.sqrt(port_var)

        m1, m2 = st.columns(2)
        m1.metric("Portfolio Annualized Return",    f"{port_ret:.2%}")
        m2.metric("Portfolio Annualized Volatility", f"{port_vol:.2%}")

        # Full volatility curve across all weight combinations (0% → 100%)
        weights_range = np.linspace(0, 1, 101)
        curve_vols = np.sqrt(np.maximum(
            weights_range**2       * ann_vol_a**2
            + (1 - weights_range)**2 * ann_vol_b**2
            + 2 * weights_range * (1 - weights_range) * ann_cov,
            0
        ))

        fig_pe = go.Figure()

        # The full diversification curve
        fig_pe.add_trace(
            go.Scatter(
                x=weights_range * 100,
                y=curve_vols,
                mode="lines",
                name="Portfolio Volatility",
                line=dict(color="steelblue", width=2.5),
            )
        )

        # Current slider position (red dot)
        fig_pe.add_trace(
            go.Scatter(
                x=[w_a_pct],
                y=[port_vol],
                mode="markers",
                name=f"Current ({w_a_pct}% / {w_b_pct}%)",
                marker=dict(color="red", size=13, symbol="circle"),
            )
        )

        # 100% Stock B endpoint
        fig_pe.add_trace(
            go.Scatter(
                x=[0], y=[ann_vol_b],
                mode="markers+text",
                text=[f"100% {port_b}"],
                textposition="top right",
                name=f"100% {port_b}",
                marker=dict(color="green", size=11, symbol="diamond"),
            )
        )

        # 100% Stock A endpoint
        fig_pe.add_trace(
            go.Scatter(
                x=[100], y=[ann_vol_a],
                mode="markers+text",
                text=[f"100% {port_a}"],
                textposition="top left",
                name=f"100% {port_a}",
                marker=dict(color="darkorange", size=11, symbol="diamond"),
            )
        )

        fig_pe.update_layout(
            title=f"Annualized Portfolio Volatility vs. Weight in {port_a}",
            xaxis_title=f"Weight in {port_a} (%)",
            yaxis_title="Annualized Volatility",
            yaxis_tickformat=".0%",
            height=450,
        )
        st.plotly_chart(fig_pe, use_container_width=True)

        corr_ab = returns[port_a].corr(returns[port_b])
        st.caption(
            f"Current correlation between {port_a} and {port_b}: **{corr_ab:.3f}**. "
            "When ρ < 1, the curve dips below both individual volatilities — "
            "this is the diversification benefit."
        )

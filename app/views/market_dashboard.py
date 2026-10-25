"""Market Dashboard: S&P 500 and VIX history from the frozen snapshot (D10-D12)."""

import html

import pandas as pd
import streamlit as st

import ui
from fincrisis.dashboard import charts, disclosures

EXPLANATION_RED = "#e8918b" if ui.dark_mode() else "#a94442"  # muted red, at least 4.5:1 contrast in either theme


def explain(col, text: str) -> None:
    """A small muted-red explanation beneath an indicator, with room above and below."""
    col.caption(f'<div style="color: {EXPLANATION_RED}; margin: 0.5rem 0 1.25rem; line-height: 1.45;">'
                f"{html.escape(text)}</div>", unsafe_allow_html=True)

snap = ui.page_header("Market Dashboard",
                      "Explore historical S&P 500 prices and market volatility, along with indicators used by "
                      "our AI models to analyze market stress.")

if snap is not None:
    spx, vix = snap["spx"], snap["vix"]
    features = ui.features()
    cols = st.columns(4)
    day_change = spx.iloc[-1] / spx.iloc[-2] - 1
    cols[0].metric(f"S&P 500 close, {spx.index[-1].date()}", f"{spx.iloc[-1]:,.2f}",
                   f"{day_change:+.2%} on the day", delta_color="off")
    explain(cols[0], "Tracks 500 large US companies; this is its value when trading ended that day. "
                     f"{day_change:+.2%} means it {'fell' if day_change < 0 else 'rose'} by "
                     f"{abs(day_change):.2%} from the day before.")
    cols[1].metric(f"VIX close, {vix.index[-1].date()}", f"{vix.iloc[-1]:.2f}",
                   f"{vix.iloc[-1] - vix.iloc[-2]:+.2f} points", delta_color="off")
    explain(cols[1], "How much investors expect US stocks to move over the next 30 days or so. Higher means "
                     "more uncertainty expected, not that a crash is coming.")
    if features is not None:
        last = features.dropna().iloc[-1]
        cols[2].metric("Below 252-session high", f"{last['drawdown_252']:.1%}")
        explain(cols[2], "Compares the S&P 500 today with its highest close of roughly the past year. "
                         f"{last['drawdown_252']:.1%} means it is {abs(last['drawdown_252']):.1%} below that high.")
        cols[3].metric("20-day downside volatility", f"{last['downside_vol_20']:.1%}", help="Annualized (D12).")
        explain(cols[3], "How large the market's daily drops have been over the last 20 trading days, scaled to a "
                         f"yearly rate. Higher means rougher down days; {last['downside_vol_20']:.1%} is not a "
                         "forecast of a fall that size.")

    # Hide the Zoom, Pan, Autoscale and Reset axes buttons; hover tooltips are unaffected.
    toolbar = {"modeBarButtonsToRemove": ["zoom2d", "pan2d", "autoScale2d", "resetScale2d"]}
    closes = pd.DataFrame({"S&P 500 close": spx, "VIX close": vix}).rename_axis("date").reset_index()
    ui.figure(charts.price_and_vix(snap), table=closes.sort_values("date", ascending=False), key="price",
              config=toolbar)
    if features is not None:
        table = (features[["drawdown_252", "downside_vol_20"]].dropna().rename_axis("date").reset_index()
                 .rename(columns={"drawdown_252": "drawdown from 252-session high",
                                  "downside_vol_20": "20-day downside volatility"}))
        ui.chart(charts.drawdown_and_volatility, features, table=table.sort_values("date", ascending=False),
                 key="drawdown", config=toolbar)

    overview = disclosures.market_overview(spx, vix, features)
    if overview:
        with st.container(border=True):
            st.subheader("Market overview summary")
            st.markdown(" ".join(overview))
            st.caption("Each comparison is with the market's own history in the snapshot; it describes what has "
                       "happened, not what will happen.")

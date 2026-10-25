"""Market Dashboard: S&P 500 and VIX history from the frozen snapshot (D10-D12)."""

import pandas as pd
import streamlit as st

import ui
from fincrisis.dashboard import charts, disclosures

snap = ui.page_header("Market Dashboard",
                      "Explore historical S&P 500 prices and market volatility, along with indicators used by "
                      "our AI models to analyze market stress.",
                      research_notice=False)

if snap is not None:
    spx, vix = snap["spx"], snap["vix"]
    features = ui.features()
    cols = st.columns(4)
    cols[0].metric(f"S&P 500 close, {spx.index[-1].date()}", f"{spx.iloc[-1]:,.2f}",
                   f"{spx.iloc[-1] / spx.iloc[-2] - 1:+.2%} on the day", delta_color="off")
    cols[1].metric(f"VIX close, {vix.index[-1].date()}", f"{vix.iloc[-1]:.2f}",
                   f"{vix.iloc[-1] - vix.iloc[-2]:+.2f} points", delta_color="off")
    if features is not None:
        last = features.dropna().iloc[-1]
        cols[2].metric("Below 252-session high", f"{last['drawdown_252']:.1%}")
        cols[3].metric("20-day downside volatility", f"{last['downside_vol_20']:.1%}", help="Annualized (D12).")

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

    with st.expander("Snapshot details"):
        st.markdown(f"- Source: {snap['source']}\n- Downloaded: {snap['downloaded_at_utc']} (UTC)\n"
                    f"- Last completed trading day in both series: {snap['as_of']}")
        st.dataframe(pd.DataFrame(snap["series"]).T, width="stretch")
    ui.limitations(disclosures.market_data_limitations(snap))
else:
    ui.limitations(["The market snapshot could not be loaded, so nothing is shown."])

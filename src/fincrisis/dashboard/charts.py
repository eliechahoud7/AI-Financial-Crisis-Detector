"""Plotly figure builders for the dashboard (docs/decisions.md, D20).

Builders take what the dashboard loaders return (`loaders.py`) and return a
`plotly.graph_objects.Figure`; they never read files, fit models or import
Streamlit. Every number shown comes from the frozen outputs passed in. A
builder raises `ChartDataError` when its input lacks what it needs;
`build_or_message` turns that into a figure that states the problem.

Visual conventions, the same in every chart:
- historical observations (prices, observed label-1 periods) are neutral ink
  or gray shading;
- model scores are colored lines, one fixed color per model (`MODEL_COLORS`),
  and their axes say they are scores, not probabilities;
- the frozen threshold is a dashed line, and days at or above it are marked
  as "signal" with the status color and a label, never by color alone;
- two measures of different scale get separate panels, never a second axis;
- legends sit below the plot, so they never cover the title.

Figures are built in light colors; `themed(fig, dark=True)` swaps in the
palette's dark-mode steps.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve

# Validated categorical slots (dataviz reference palette, slots 1-4, light mode;
# adjacent CVD separation >= 9.1, normal-vision >= 22.9). Slots 3 and 4 are
# below 3:1 contrast on the light surface, so every chart has a legend and hover.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REFERENCE = "#898781"   # muted ink: dummy baselines and base rates
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIGNAL = "#d03b3b"       # status "critical": always shipped with a text label
OBSERVED = "rgba(137, 135, 129, 0.18)"  # gray shading for observed label-1 periods

# Color follows the model, on every page. Company-distress families reuse the
# colors of the market models of the same kind.
MODEL_COLORS = {
    "rule_downside_vol_20": SERIES[0], "xgboost": SERIES[1], "logistic_regression": SERIES[2],
    "rule_vix_pct_252": SERIES[3], "dummy": REFERENCE,
}
MODEL_LABELS = {
    "rule_downside_vol_20": "Rule: 20-day downside volatility", "rule_vix_pct_252": "Rule: VIX 252-day percentile",
    "logistic_regression": "Logistic regression", "xgboost": "XGBoost", "dummy": "Dummy (base rate)",
}
SCORE_AXIS = "Model score (ranks risk; not a probability)"
HOVER_BG = "#ffffff"
HEAT_LOW, HEAT_HIGH = "#cde2fb", "#6da7ec"   # sequential blue steps 100 and 300
DRAWDOWN_FILL = "rgba(82, 81, 78, 0.15)"

# Light -> dark steps of the same palette (dataviz reference: dark categorical slots and chart chrome).
DARK = {
    "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500",
    INK: "#f0efec", INK_2: "#c3c2b7", GRID: "#2c2c2a", AXIS: "#383835", SURFACE: "#1a1a19", HOVER_BG: "#262624",
    HEAT_LOW: "#184f95", HEAT_HIGH: "#3987e5", DRAWDOWN_FILL: "rgba(195, 194, 183, 0.18)",
    OBSERVED: "rgba(137, 135, 129, 0.32)",
}


class ChartDataError(ValueError):
    """The input lacks the data a chart needs."""


def _require(mapping, *keys, what: str = "input"):
    if mapping is None:
        raise ChartDataError(f"{what} is missing")
    missing = [k for k in keys if k not in mapping or mapping[k] is None]
    if missing:
        raise ChartDataError(f"{what} lacks {', '.join(map(str, missing))}")


def _columns(frame, *columns, what: str):
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ChartDataError(f"{what} is missing or empty")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ChartDataError(f"{what} lacks columns {', '.join(missing)}")


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def color(model: str, position: int = 0) -> str:
    return MODEL_COLORS.get(model, SERIES[position % len(SERIES)])


def _style(fig: go.Figure, title: str, height: int = 420, subtitle: str | None = None) -> go.Figure:
    text = f"{title}<br><sup>{subtitle}</sup>" if subtitle else title
    legend = any(getattr(trace, "showlegend", None) is not False for trace in fig.data) and fig.layout.showlegend is not False
    fig.update_layout(
        title={"text": text, "x": 0.01, "xanchor": "left", "xref": "container", "y": 1, "yref": "container",
               "yanchor": "top", "pad": {"t": 14}, "font": {"size": 16, "color": INK}},
        template="plotly_white", height=height + (60 if legend else 0), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK_2, "size": 12},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0},
        margin={"l": 60, "r": 20, "t": (90 if subtitle else 60) + (20 if fig.layout.annotations else 0),
                "b": 120 if legend else 50},
        hoverlabel={"font": {"color": INK}, "bgcolor": HOVER_BG},
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def themed(fig: go.Figure, dark: bool) -> go.Figure:
    """The figure in dark-mode colors (same hues, dark-surface steps), or unchanged for light mode."""
    if not dark:
        return fig
    pattern = re.compile("|".join(re.escape(k) for k in sorted(DARK, key=len, reverse=True)), re.IGNORECASE)
    swapped = pattern.sub(lambda m: DARK[next(k for k in DARK if k.lower() == m.group(0).lower())], fig.to_json())
    return go.Figure(json.loads(swapped))


def message_figure(message: str, title: str = "Chart unavailable") -> go.Figure:
    """An empty figure that states why the chart cannot be drawn."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"size": 13, "color": INK_2})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title, height=260)


def build_or_message(builder, *args, **kwargs) -> go.Figure:
    """Call `builder`; if its input is missing or invalid, return a figure that explains why instead."""
    try:
        return builder(*args, **kwargs)
    except ChartDataError as error:
        return message_figure(str(error), title=f"{builder.__name__.replace('_', ' ').capitalize()}: unavailable")


def _reference(fig: go.Figure, y: float, name: str, row: int | None = None, col: int | None = None,
               legend: bool = True) -> None:
    """A dashed reference line (base rate, random ranking), named in the legend rather than written over the data."""
    fig.add_hline(y=y, line={"color": REFERENCE, "dash": "dash", "width": 1.5}, row=row, col=col)
    if legend:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                 line={"color": REFERENCE, "dash": "dash", "width": 1.5}), row=row, col=col)


def _wrap(name: str) -> str:
    """Break a long axis label onto two lines, so tick labels need not be rotated."""
    return name.replace(": ", ":<br>").replace(" (", "<br>(")


def _episodes(dates: pd.Series, labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of label 1 as (first date, last date)."""
    runs, start, previous = [], None, None
    for day, y in zip(pd.to_datetime(dates), labels):
        if y == 1 and start is None:
            start = day
        elif y != 1 and start is not None:
            runs.append((start, previous))
            start = None
        previous = day
    if start is not None:
        runs.append((start, previous))
    return runs


def _break_gaps(frame: pd.DataFrame, column: str, max_days: int = 10) -> pd.DataFrame:
    """Insert an empty row wherever consecutive dates are more than `max_days` apart, so lines do not bridge gaps."""
    gaps = frame[column].diff().dt.days.gt(max_days).to_numpy()
    if not gaps.any():
        return frame
    blanks = frame[gaps].assign(**{c: np.nan for c in frame.columns if c != column})
    blanks[column] = blanks[column] - pd.Timedelta(days=1)
    return pd.concat([frame, blanks]).sort_values(column, kind="stable")


# ----------------------------------------------------------------------------- market history

def price_and_vix(snapshot: dict) -> go.Figure:
    """S&P 500 and VIX closes from the frozen snapshot, in two panels with a shared date axis."""
    _require(snapshot, "spx", "vix", "as_of", what="market snapshot")
    spx, vix = snapshot["spx"], snapshot["vix"]
    if len(spx) == 0 or len(vix) == 0:
        raise ChartDataError("market snapshot has no prices")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.62, 0.38],
                        subplot_titles=("S&P 500 close (index points, log scale)", "VIX close (index points)"))
    fig.add_trace(go.Scatter(x=spx.index, y=spx.values, name="S&P 500 close", line={"color": INK, "width": 1.5},
                             hovertemplate="%{x|%Y-%m-%d}<br>S&P 500 close %{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=vix.index, y=vix.values, name="VIX close", line={"color": INK_2, "width": 1.5},
                             hovertemplate="%{x|%Y-%m-%d}<br>VIX close %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(type="log", title_text="Index level", row=1, col=1)
    fig.update_yaxes(title_text="VIX", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(rangeselector={"buttons": [
        {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
        {"count": 5, "label": "5y", "step": "year", "stepmode": "backward"},
        {"step": "all", "label": "All"}], "bgcolor": GRID, "activecolor": AXIS, "font": {"color": INK}}, row=1, col=1)
    fig.update_layout(hovermode="x unified", showlegend=False)
    return _style(fig, "S&P 500 and VIX: historical closes", height=560,
                  subtitle=f"Frozen snapshot to {snapshot['as_of']}; observations, not model output")


def drawdown_and_volatility(features: pd.DataFrame) -> go.Figure:
    """Drawdown from the 252-session high and 20-day downside volatility, both backward-looking (D12)."""
    _columns(features, "drawdown_252", "downside_vol_20", what="market features")
    frame = features[["drawdown_252", "downside_vol_20"]].dropna()
    if frame.empty:
        raise ChartDataError("market features have no complete rows")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Drawdown from the 252-session high", "20-day downside volatility (annualized)"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["drawdown_252"], name="Drawdown", fill="tozeroy",
                             line={"color": INK_2, "width": 1}, fillcolor=DRAWDOWN_FILL,
                             hovertemplate="%{x|%Y-%m-%d}<br>Drawdown %{y:.1%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["downside_vol_20"], name="Downside volatility",
                             line={"color": SERIES[0], "width": 1.5},
                             hovertemplate="%{x|%Y-%m-%d}<br>Downside volatility %{y:.1%}<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="Below high", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="Volatility", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(hovermode="x unified", showlegend=False)
    last = frame.index[-1].date()
    return _style(fig, "Drawdown and downside volatility", height=520,
                  subtitle=f"Computed from closes up to each day, to {last}")


# ----------------------------------------------------------------------------- market-stress scores

def stress_score_timeline(results: dict, model: str | None = None, current: dict | None = None) -> go.Figure:
    """One model's walk-forward validation (2000-2014) and held-out test scores against its frozen threshold.

    Gray bands are observed label-1 periods (the S&P 500 fell at least 5% below its running high within the
    next 20 sessions); they are known only afterwards. Red markers are days at or above the frozen threshold.
    `current` (from `loaders.current_market_signal`) adds the latest snapshot day, which has no label yet.
    """
    _require(results, "oof_predictions", "test_predictions", "thresholds", "primary_model", what="market-stress results")
    model = model or results["primary_model"]
    if model not in results["thresholds"]:
        raise ChartDataError(f"no frozen threshold for model {model!r}")
    frames = []
    for name, stage in (("oof_predictions", "Validation (walk-forward)"), ("test_predictions", "Held-out test")):
        _columns(results[name], "date", "label", model, what=name)
        frames.append(results[name][["date", "label", model]].assign(stage=stage))
    data = pd.concat(frames).sort_values("date")
    threshold = results["thresholds"][model]
    fig = go.Figure()
    for start, end in _episodes(data["date"], data["label"]):
        fig.add_vrect(x0=start, x1=end + pd.Timedelta(days=1), fillcolor=OBSERVED, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Observed label-1 period (known afterwards)",
                             marker={"symbol": "square", "size": 12, "color": OBSERVED}))
    for stage, frame in data.groupby("stage", sort=False):
        frame = _break_gaps(frame, "date")
        fig.add_trace(go.Scatter(
            x=frame["date"], y=frame[model], name=f"{label(model)} score, {stage.lower()}", mode="lines",
            line={"color": color(model), "width": 1.5, "dash": "solid" if stage == "Held-out test" else "dot"},
            customdata=np.stack([frame["label"].astype(object),
                                 np.where(frame[model] >= threshold, "yes", "no")], axis=1),
            hovertemplate=(f"%{{x|%Y-%m-%d}} ({stage})<br>Score %{{y:.4f}}<br>Signal (score ≥ {threshold:.4f}): "
                           "%{customdata[1]}<br>Observed label: %{customdata[0]}<extra></extra>")))
    signal = data[data[model] >= threshold]
    fig.add_trace(go.Scatter(x=signal["date"], y=signal[model], mode="markers", name="Signal: score ≥ frozen threshold",
                             marker={"color": SIGNAL, "size": 4}, hoverinfo="skip"))
    fig.add_hline(y=threshold, line={"color": INK, "dash": "dash", "width": 1})
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Frozen threshold {threshold:.4f}",
                             line={"color": INK, "dash": "dash", "width": 1}))
    test_start = results["test_predictions"]["date"].min()
    fig.add_vline(x=test_start, line={"color": AXIS, "width": 1})
    fig.add_annotation(x=test_start, y=1, yref="paper", text="Held-out test starts", showarrow=False,
                       xanchor="left", yanchor="bottom", font={"color": INK_2})
    if current is not None:
        _require(current, "as_of", "primary", what="current signal")
        if current["primary"]["name"] == model:
            p = current["primary"]
            state = "signal" if p["elevated"] else "no signal"
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(current["as_of"])], y=[p["score"]], mode="markers", text=[f"Latest: {state}"],
                name=f"Latest snapshot day, {current['as_of']} ({state}; no label yet)",
                marker={"color": SIGNAL if p["elevated"] else INK, "size": 10, "symbol": "diamond",
                        "line": {"color": SURFACE, "width": 2}},
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>Score %{{y:.4f}} (no label yet)<extra></extra>"))
    fig.update_yaxes(title_text=SCORE_AXIS)
    fig.update_xaxes(title_text="Date", type="date")  # the legend placeholder trace has no dates
    return _style(fig, f"{label(model)}: scores against the frozen threshold", height=480,
                  subtitle="Gray: observed stress periods, known only afterwards. Red: days at or above the threshold")


# ----------------------------------------------------------------------------- market-stress performance

def _stage(results: dict, stage: str) -> tuple[dict, str]:
    if stage not in ("test", "validation"):
        raise ChartDataError(f"unknown stage {stage!r}; use 'test' or 'validation'")
    key = "test_metrics" if stage == "test" else "cv_metrics"
    _require(results, key, what="market-stress results")
    metrics = results[key]
    _require(metrics, "models", "prevalence", what=key)
    return metrics, ("held-out test" if stage == "test" else "walk-forward validation")


def stress_model_metrics(results: dict, stage: str = "test") -> go.Figure:
    """Average precision of every market-stress model, with block-bootstrap 95% intervals, against the base rate."""
    metrics, name = _stage(results, stage)
    intervals = metrics.get("bootstrap", {}).get("primary", {}).get("average_precision", {})
    models = list(metrics["models"])
    values = [metrics["models"][m]["average_precision"] for m in models]
    lo = [values[i] - intervals[m][0] if m in intervals else 0 for i, m in enumerate(models)]
    hi = [intervals[m][1] - values[i] if m in intervals else 0 for i, m in enumerate(models)]
    primary = results.get("primary_model")
    names = [_wrap(label(m) + (" (primary)" if m == primary else "")) for m in models]
    fig = go.Figure(go.Bar(
        x=names, y=values, marker={"color": [color(m, i) for i, m in enumerate(models)], "line": {"color": SURFACE, "width": 2}},
        error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo, "color": INK_2, "thickness": 1},
        customdata=[[intervals.get(m, [np.nan, np.nan])[0], intervals.get(m, [np.nan, np.nan])[1]] for m in models],
        hovertemplate="%{x}<br>Average precision %{y:.3f}<br>95% interval %{customdata[0]:.3f} to %{customdata[1]:.3f}"
                      "<extra></extra>", showlegend=False))
    _reference(fig, metrics["prevalence"], f"Base rate {metrics['prevalence']:.3f} (expected AP of a random score)")
    fig.update_yaxes(title_text="Average precision", range=[0, 1])
    fig.update_xaxes(title_text="Model", tickangle=0)
    return _style(fig, "Market-stress models: average precision",
                  subtitle=f"{name.capitalize()}, {metrics.get('rows', '?')} days; bars show 95% block-bootstrap intervals")


def stress_per_year(results: dict, stage: str = "test", models: list[str] | None = None) -> go.Figure:
    """Average precision per calendar year, with each year's base rate; years flagged unreliable are hollow."""
    metrics, name = _stage(results, stage)
    models = models or [m for m in metrics["models"] if m != "dummy"]
    fig = go.Figure()
    prevalence = None
    for i, m in enumerate(models):
        if m not in metrics["models"] or not metrics["models"][m].get("per_year"):
            raise ChartDataError(f"{stage} metrics have no per-year results for {m!r}")
        rows = pd.DataFrame(metrics["models"][m]["per_year"])
        _columns(rows, "year", "average_precision", "prevalence", "status", what=f"per-year results of {m}")
        prevalence = rows[["year", "prevalence"]]
        reliable = rows["status"].eq("ok")
        fig.add_trace(go.Scatter(
            x=rows["year"], y=rows["average_precision"], name=label(m), mode="lines+markers",
            line={"color": color(m, i), "width": 2},
            marker={"size": 8, "color": [color(m, i) if r else SURFACE for r in reliable],
                    "line": {"color": color(m, i), "width": 2}},
            customdata=np.stack([rows["status"], rows.get("label_1", pd.Series([np.nan] * len(rows)))], axis=1),
            hovertemplate="%{x}: average precision %{y:.3f}<br>Label-1 days %{customdata[1]}<br>Status %{customdata[0]}"
                          f"<extra>{label(m)}</extra>"))
    fig.add_trace(go.Scatter(x=prevalence["year"], y=prevalence["prevalence"], name="Base rate (share of label-1 days)",
                             mode="lines", line={"color": REFERENCE, "dash": "dash", "width": 1.5},
                             hovertemplate="%{x}: base rate %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Average precision", range=[0, 1])
    fig.update_xaxes(title_text="Year", dtick=1)
    return _style(fig, "Market-stress models: average precision by year",
                  subtitle=f"{name.capitalize()}; hollow markers: too few label-1 days to be reliable")


def stress_calibration(results: dict, stage: str = "test") -> go.Figure:
    """Mean score against observed label-1 rate per score bin, for the probability-shaped models only."""
    metrics, name = _stage(results, stage)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line={"color": REFERENCE, "dash": "dash", "width": 1}, hoverinfo="skip"))
    shown = 0
    for i, (m, values) in enumerate(metrics["models"].items()):
        bins = values.get("calibration")
        if not bins:  # rule scores are not probabilities, so they have no calibration table
            continue
        rows = pd.DataFrame(bins)
        _columns(rows, "bin", "n", "mean_predicted", "observed_rate", what=f"calibration of {m}")
        shown += 1
        fig.add_trace(go.Scatter(
            x=rows["mean_predicted"], y=rows["observed_rate"], name=label(m), mode="lines+markers",
            line={"color": color(m, i), "width": 2}, marker={"size": 9},
            customdata=np.stack([rows["bin"], rows["n"]], axis=1),
            hovertemplate="Score bin %{customdata[0]} (%{customdata[1]} days)<br>Mean score %{x:.3f}<br>"
                          f"Observed label-1 rate %{{y:.3f}}<extra>{label(m)}</extra>"))
    if not shown:
        raise ChartDataError(f"no model has a calibration table in the {name} metrics")
    fig.update_xaxes(title_text="Mean model score in bin", range=[0, 1])
    fig.update_yaxes(title_text="Observed share of label-1 days", range=[0, 1])
    return _style(fig, f"Calibration on the {name}",
                  subtitle="Points off the diagonal mean the scores are not probabilities")


# ----------------------------------------------------------------------------- shared evaluation charts

def confusion_matrix(counts: dict, title: str, positive: str = "Label 1", negative: str = "Label 0",
                     subtitle: str | None = None) -> go.Figure:
    """Counts at one frozen threshold (`tn`, `fp`, `fn`, `tp`), with each cell's share of its observed class."""
    _require(counts, "tn", "fp", "fn", "tp", what="threshold results")
    tn, fp, fn, tp = (int(counts[k]) for k in ("tn", "fp", "fn", "tp"))
    z = np.array([[tn, fp], [fn, tp]])
    totals = z.sum(axis=1, keepdims=True)
    share = np.divide(z, totals, out=np.zeros_like(z, dtype=float), where=totals > 0)
    text = [[f"<b>{z[r, c]:,}</b><br>{share[r, c]:.0%} of row" for c in range(2)] for r in range(2)]
    hover = [[f"{z[r, c]:,} cases: {share[r, c]:.0%} of all observed {(negative, positive)[r]}" for c in range(2)]
             for r in range(2)]
    fig = go.Figure(go.Heatmap(
        z=share, x=[f"Predicted<br>{negative}", f"Predicted<br>{positive}"],
        y=[f"Observed<br>{negative}", f"Observed<br>{positive}"],
        text=text, texttemplate="%{text}", textfont={"size": 14}, customdata=hover,
        colorscale=[[0, HEAT_LOW], [1, HEAT_HIGH]], zmin=0, zmax=1,
        showscale=False, hovertemplate="%{customdata}<extra></extra>", xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed", title_text="Observed")
    fig.update_xaxes(title_text="Predicted", side="bottom", tickangle=0)
    return _style(fig, title, height=380, subtitle=subtitle)


def stress_confusion(results: dict, model: str | None = None) -> go.Figure:
    """Held-out test counts for one market-stress model at its frozen max-F1 threshold."""
    metrics, _ = _stage(results, "test")
    model = model or results.get("primary_model")
    if model not in metrics["models"]:
        raise ChartDataError(f"no test results for model {model!r}")
    at = metrics["models"][model].get("at_max_f1_threshold")
    _require(at, "threshold", what=f"threshold results of {model}")
    return confusion_matrix(at, f"{label(model)}: held-out test days",
                            positive="stress (label 1)", negative="no stress (label 0)",
                            subtitle=f"At the frozen threshold {at['threshold']:.4f}")


def precision_recall(predictions: pd.DataFrame, columns: dict[str, str], recorded: dict[str, dict], title: str,
                     label_column: str = "label") -> go.Figure:
    """Precision-recall curves from saved predictions, with each model's recorded operating point.

    `columns` maps model -> prediction column. `recorded` maps model -> its frozen test results: the legend
    shows its `average_precision`, and the marker its `at_max_f1_threshold` precision and recall, as recorded.
    The curve itself is traced from the same saved predictions; nothing is re-estimated.
    """
    _columns(predictions, label_column, *columns.values(), what="saved predictions")
    y = predictions[label_column].to_numpy()
    if len(np.unique(y)) < 2:
        raise ChartDataError("saved predictions contain only one class")
    fig = go.Figure()
    for i, (model, column) in enumerate(columns.items()):
        precision, recall, _ = precision_recall_curve(y, predictions[column].to_numpy())
        result = recorded.get(model, {})
        name = f"{label(model)} (AP {result['average_precision']:.3f})" if "average_precision" in result else label(model)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name, line={"color": color(model, i), "width": 2},
                                 hovertemplate=f"Recall %{{x:.2f}}<br>Precision %{{y:.2f}}<extra>{label(model)}</extra>"))
        at = result.get("at_max_f1_threshold")
        if at:
            fig.add_trace(go.Scatter(x=[at["recall"]], y=[at["precision"]], mode="markers", showlegend=False,
                                     marker={"color": color(model, i), "size": 11, "symbol": "x",
                                             "line": {"color": SURFACE, "width": 2}},
                                     hovertemplate=f"{label(model)} at frozen threshold {at['threshold']:.4f}<br>"
                                                   "Recall %{x:.2f}, precision %{y:.2f}<extra></extra>"))
    base = float(np.mean(y))
    _reference(fig, base, f"Base rate {base:.3f}")
    fig.update_xaxes(title_text="Recall (share of label-1 cases flagged)", range=[0, 1])
    fig.update_yaxes(title_text="Precision (share of flags that were label 1)", range=[0, 1.02])
    return _style(fig, title, subtitle="× marks each model's recorded operating point at its frozen threshold")


def stress_precision_recall(results: dict) -> go.Figure:
    metrics, _ = _stage(results, "test")
    _require(results, "test_predictions", "thresholds", what="market-stress results")
    models = [m for m in metrics["models"] if m != "dummy"]
    return precision_recall(results["test_predictions"], {m: m for m in models}, metrics["models"],
                            "Market stress: precision and recall, held-out test")


# ----------------------------------------------------------------------------- company distress

def _families(results: dict) -> dict:
    _require(results, "test_metrics", what="company-distress results")
    _require(results["test_metrics"], "results", "test_rows", "test_label_1", what="company-distress test metrics")
    return results["test_metrics"]["results"]


def distress_model_metrics(results: dict) -> go.Figure:
    """Test average precision and ROC-AUC per model family, with bootstrap 95% intervals, in two panels."""
    families = _families(results)
    test = results["test_metrics"]
    prevalence = test["test_label_1"] / test["test_rows"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average precision", "ROC-AUC"), horizontal_spacing=0.12)
    for col, metric, reference, ref_text in ((1, "average_precision", prevalence, f"Base rate {prevalence:.3f}"),
                                             (2, "roc_auc", 0.5, "Random ranking 0.5")):
        names, values, lo, hi, colors = [], [], [], [], []
        for i, (family, r) in enumerate(families.items()):
            ci = r.get("bootstrap_95ci_at_max_f1", {}).get(metric, [r[metric], r[metric]])
            names.append(f"{_wrap(label(family))}<br>({r['config']})")
            values.append(r[metric])
            lo.append(r[metric] - ci[0])
            hi.append(ci[1] - r[metric])
            colors.append(color(family, i))
        fig.add_trace(go.Bar(x=names, y=values, marker={"color": colors, "line": {"color": SURFACE, "width": 2}},
                             error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo, "color": INK_2},
                             customdata=np.stack([np.array(values) - lo, np.array(values) + hi], axis=1),
                             hovertemplate="%{x}<br>%{y:.3f} (95% interval %{customdata[0]:.3f} to %{customdata[1]:.3f})"
                                           "<extra></extra>", showlegend=False), row=1, col=col)
        _reference(fig, reference, ref_text, row=1, col=col)
        fig.update_yaxes(range=[0, 1], row=1, col=col)
        fig.update_xaxes(tickangle=0, row=1, col=col)
    return _style(fig, "Company-distress models on the held-out test", height=440,
                  subtitle=f"{test['test_rows']} companies, {test['test_label_1']} with label 1 (label meaning unverified); "
                           "bars show bootstrap 95% intervals")


def distress_confusion(results: dict, family: str = "xgboost", at: str = "max_f1") -> go.Figure:
    """Held-out test counts for one family at its frozen max-F1 threshold (`at="max_f1"`) or at 0.5."""
    families = _families(results)
    if family not in families:
        raise ChartDataError(f"no test results for family {family!r}")
    key = {"max_f1": "at_max_f1_threshold", "0.5": "at_0.5"}.get(at)
    if key is None:
        raise ChartDataError(f"unknown operating point {at!r}; use 'max_f1' or '0.5'")
    counts = families[family].get(key)
    _require(counts, "threshold", what=f"{key} results of {family}")
    which = "the frozen max-F1 threshold" if at == "max_f1" else "0.5"
    return confusion_matrix(counts, f"{label(family)}: held-out test companies", positive="label 1",
                            negative="label 0", subtitle=f"At {which} ({counts['threshold']:.3f}); label meaning unverified")


def distress_precision_recall(results: dict) -> go.Figure:
    families = _families(results)
    _require(results, "test_predictions", what="company-distress results")
    chosen = {f: r["config"] for f, r in families.items() if f != "dummy"}
    return precision_recall(results["test_predictions"], chosen, families,
                            "Company distress: precision and recall, held-out test")


def distress_cv_folds(results: dict) -> go.Figure:
    """Average precision of every evaluated configuration in each cross-validation fold; selected ones are colored."""
    _require(results, "cv_results", what="company-distress results")
    cv = results["cv_results"]
    _columns(cv, "config", "average_precision", what="cv_results.csv")
    selected = {r["config"]: f for f, r in _families(results).items()}
    order = cv.groupby("config")["average_precision"].mean().sort_values().index
    fig = go.Figure()
    for config in order:
        values = cv.loc[cv["config"] == config, "average_precision"]
        family = selected.get(config)
        fig.add_trace(go.Box(
            x=values, y=[config] * len(values), name=config + (f" (selected {label(family)})" if family else ""),
            orientation="h", boxpoints="all", jitter=0.4, pointpos=0, showlegend=False,
            marker={"color": color(family) if family else REFERENCE, "size": 6}, line={"color": color(family) if family else REFERENCE},
            fillcolor="rgba(0,0,0,0)",
            hovertemplate=f"{config}<br>Fold average precision %{{x:.3f}}<extra></extra>"))
    fig.update_xaxes(title_text="Average precision in one validation fold", range=[0, 1])
    fig.update_yaxes(title_text="Configuration")
    folds = int(cv.groupby("config").size().max())
    return _style(fig, "Company-distress cross-validation", height=140 + 40 * len(order),
                  subtitle=f"{folds} folds per configuration; colored: the configuration selected for each family")


def distress_group_importance(results: dict) -> go.Figure:
    """Drop in fold average precision when the large-value feature group is shuffled together (D9).

    This is the only importance the pipeline saved: one group of columns, not individual features.
    """
    _require(results, "permutation_importance", what="company-distress results")
    importance = results["permutation_importance"]
    _columns(importance, "config", "fold", "mean_drop", "base_average_precision", what="permutation_importance.csv")
    group = (results.get("selection") or {}).get("large_value_columns") or []
    family_of = {r["config"]: f for f, r in _families(results).items()}
    fig = go.Figure()
    for i, (config, rows) in enumerate(importance.groupby("config", sort=False)):
        family = family_of.get(config)
        fig.add_trace(go.Bar(
            x=rows["fold"].astype(str), y=rows["mean_drop"], name=f"{config}" + (f" ({label(family)})" if family else ""),
            marker={"color": color(family, i) if family else SERIES[i % 4], "line": {"color": SURFACE, "width": 2}},
            customdata=rows[["base_average_precision"]].to_numpy(),
            hovertemplate="Fold %{x}<br>Average precision drop %{y:.3f}<br>Fold average precision before shuffling "
                          f"%{{customdata[0]:.3f}}<extra>{config}</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Cross-validation fold")
    fig.update_yaxes(title_text="Drop in average precision")
    size = f"{len(group)} large-value columns" if group else "the large-value columns"
    return _style(fig, "Company distress: grouped permutation importance",
                  subtitle=f"Effect of shuffling {size} together in each fold (grouped, not per feature)")

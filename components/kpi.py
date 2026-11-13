import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ── Helpers ───────────────────────────────────────────────────────────────────

from components.utils import get_num_cols, get_cat_cols

def _num_cols(df):
    return get_num_cols(df)

def _cat_cols(df):
    return get_cat_cols(df)


def _sparkline(values: list, color: str = "#00D4FF") -> go.Figure:
    """Tiny inline sparkline — no axes, no labels, transparent bg."""
    y = list(values)
    fig = go.Figure(go.Scatter(
        y=y, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)" if color.startswith("#") else color,
    ))
    fig.update_layout(
        height=60, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _trend_arrow(current, previous):
    if previous == 0:
        return "—", "#94A3B8"
    pct = (current - previous) / abs(previous) * 100
    if pct > 0:
        return f"▲ {pct:.1f}%", "#10B981"
    elif pct < 0:
        return f"▼ {abs(pct):.1f}%", "#EF4444"
    return "— 0.0%", "#94A3B8"


def _fmt(val):
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.1f}K"
    return f"{val:.2f}"


# ── KPI Card renderer ─────────────────────────────────────────────────────────

def _kpi_card(col_ctx, icon, label, value_str, sub, trend_str, trend_color,
              spark_vals=None, spark_color="#00D4FF", border_color="var(--cyan)"):
    with col_ctx:
        trend_html = (
            f'<span style="color:{trend_color};font-size:12px;font-weight:600">{trend_str}</span>'
            if trend_str else ""
        )
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color:{border_color}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value_str}</div>
          <div class="kpi-sub" style="display:flex;align-items:center;gap:8px">
            <span style="color:#4B5563">{sub}</span> {trend_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

        if spark_vals and len(spark_vals) > 2:
            st.plotly_chart(
                _sparkline(spark_vals, spark_color),
                use_container_width=True,
                config={"staticPlot": True, "displayModeBar": False},
                key=f"spark_{label}",
            )


# ── Main component ────────────────────────────────────────────────────────────

def kpi_dashboard(df: pd.DataFrame):
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
      <div class="section-label-dot"></div>
      <span class="section-label-text">Key Metrics</span>
    </div>
    <div class="section-title">KPI Dashboard</div>
    <div class="section-desc">Auto-detected key performance indicators from your dataset.</div>
    """, unsafe_allow_html=True)

    nums = _num_cols(df)
    cats = _cat_cols(df)

    if not nums:
        st.info("No numeric columns found — KPI cards require at least one numeric column.")
        return

    # ── Row 1: Dataset-level KPIs ─────────────────────────────────────────────
    st.markdown("##### 📌 Dataset Overview")
    r1 = st.columns(4)

    total_rows  = len(df)
    total_cols  = len(df.columns)
    missing_pct = round(df.isnull().sum().sum() / (total_rows * total_cols) * 100, 1)
    complete_pct = round((1 - df.isnull().sum().sum() / (total_rows * total_cols)) * 100, 1)

    _kpi_card(r1[0], "🗂️", "Total Records",   f"{total_rows:,}",
              f"{total_cols} columns", "", "",
              border_color="#00D4FF")
    _kpi_card(r1[1], "✅", "Completeness",    f"{complete_pct}%",
              "data filled", "", "",
              border_color="#10B981" if complete_pct > 90 else "#F59E0B")
    _kpi_card(r1[2], "📐", "Numeric Columns", f"{len(nums)}",
              f"{len(cats)} categorical", "", "",
              border_color="#7C3AED")
    _kpi_card(r1[3], "⚠️", "Missing Rate",    f"{missing_pct}%",
              "of all cells", "", "",
              border_color="#EF4444" if missing_pct > 10 else "#10B981")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Per-column numeric KPIs (up to 4 columns) ─────────────────────
    st.markdown("##### 📊 Numeric Column Highlights")

    cols_to_show = nums[:4]
    r2 = st.columns(len(cols_to_show))

    palette = ["#00D4FF", "#7C3AED", "#10B981", "#F59E0B"]

    for i, col in enumerate(cols_to_show):
        series   = df[col].dropna()
        mean_val = series.mean()
        med_val  = series.median()
        std_val  = series.std()
        min_val  = series.min()
        max_val  = series.max()

        # Sparkline: sample up to 60 evenly-spaced points
        step = max(1, len(series) // 60)
        spark = series.iloc[::step].tolist()

        # Trend: first half vs second half mean
        half = len(series) // 2
        if half > 0:
            trend_str, trend_color = _trend_arrow(
                series.iloc[half:].mean(), series.iloc[:half].mean()
            )
        else:
            trend_str, trend_color = "—", "#94A3B8"

        icons  = ["📈", "📉", "🎯", "💡"]
        _kpi_card(
            r2[i],
            icons[i % len(icons)],
            col[:20],
            _fmt(mean_val),
            f"median {_fmt(med_val)}",
            trend_str, trend_color,
            spark_vals=spark,
            spark_color=palette[i % len(palette)],
            border_color=palette[i % len(palette)],
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Statistical summary cards ─────────────────────────────────────
    st.markdown("##### 🔬 Statistical Highlights")

    r3 = st.columns(4)

    # Highest variance column
    var_col = df[nums].var().idxmax()
    var_val = df[nums].var().max()

    # Most correlated pair
    if len(nums) >= 2:
        corr = df[nums].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_idx = upper.stack().idxmax()
        max_corr = upper.stack().max()
        corr_label = f"{max_idx[0][:10]} ↔ {max_idx[1][:10]}"
        corr_val_str = f"r = {max_corr:.2f}"
    else:
        corr_label = "N/A"
        corr_val_str = "Need 2+ cols"

    # Most skewed column
    skew_series = df[nums].skew().abs()
    skew_col    = skew_series.idxmax()
    skew_val    = df[nums].skew()[skew_col]
    skew_desc   = "right" if skew_val > 0 else "left"

    # Outlier count across all numeric columns
    total_outliers = 0
    for c in nums:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        total_outliers += int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())

    _kpi_card(r3[0], "📊", "Highest Variance",  var_col[:18],
              f"σ² = {_fmt(var_val)}", "", "",
              border_color="#F97316")
    _kpi_card(r3[1], "🔗", "Top Correlation",   corr_val_str,
              corr_label, "", "",
              border_color="#00D4FF")
    _kpi_card(r3[2], "📐", "Most Skewed",        f"{abs(skew_val):.2f}",
              f"{skew_col[:16]} ({skew_desc}-skewed)", "", "",
              border_color="#7C3AED")
    _kpi_card(r3[3], "🚨", "Total Outliers",     f"{total_outliers:,}",
              "across all numeric columns", "", "",
              border_color="#EF4444" if total_outliers > 0 else "#10B981")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Categorical breakdown (if available) ─────────────────────────────────
    if cats:
        st.markdown("##### 🏷️ Categorical Breakdown")
        cat_cols_show = cats[:4]
        r4 = st.columns(len(cat_cols_show))
        for i, col in enumerate(cat_cols_show):
            n_unique = df[col].nunique()
            top_val  = df[col].value_counts().index[0]
            top_pct  = round(df[col].value_counts().iloc[0] / len(df) * 100, 1)
            with r4[i]:
                st.markdown(f"""
                <div class="kpi-card" style="border-top-color:#7C3AED">
                  <div class="kpi-icon">🏷️</div>
                  <div class="kpi-label">{col[:20]}</div>
                  <div class="kpi-value" style="font-size:1.3rem">{str(top_val)[:16]}</div>
                  <div class="kpi-sub">
                    top value · {top_pct}% of rows<br>
                    <span style="color:#4B5563">{n_unique} unique categories</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
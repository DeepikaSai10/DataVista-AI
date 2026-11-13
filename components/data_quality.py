import streamlit as st
import pandas as pd
import numpy as np
from components.utils import get_num_cols, get_cat_cols


def data_quality_report(df: pd.DataFrame):
    """Renders a full Data Health Dashboard for the given DataFrame."""

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Section Header ──────────────────────────────────────
    st.markdown("""
    <div class="section-label">
      <div class="section-label-dot"></div>
      <span class="section-label-text">Data Intelligence</span>
    </div>
    <div class="section-title">Data Health Dashboard</div>
    <div class="section-desc">Automated quality scan across every column and row.</div>
    """, unsafe_allow_html=True)

    # ── Compute metrics ──────────────────────────────────────
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols

    missing_per_col = df.isnull().sum()
    total_missing   = int(missing_per_col.sum())
    missing_pct     = round(total_missing / total_cells * 100, 1) if total_cells else 0

    duplicate_rows  = int(df.duplicated().sum())
    dup_pct         = round(duplicate_rows / total_rows * 100, 1) if total_rows else 0

    num_cols   = get_num_cols(df)
    cat_cols   = get_cat_cols(df)
    date_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()

    complete_cols = int((missing_per_col == 0).sum())

    # Quality score (simple heuristic)
    score = 100
    score -= min(40, missing_pct * 2)
    score -= min(20, dup_pct * 2)
    score  = max(0, round(score))
    score_color = "#10B981" if score >= 80 else "#F59E0B" if score >= 50 else "#EF4444"
    score_label = "Excellent" if score >= 80 else "Fair" if score >= 50 else "Poor"

    # ── KPI Row ──────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    def kpi(col, icon, label, value, sub="", color="var(--cyan)"):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{color}">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    kpi(k1, "🗂️", "Total Rows",    f"{total_rows:,}",    "records loaded")
    kpi(k2, "📐", "Columns",       f"{total_cols}",       f"{complete_cols} fully complete")
    kpi(k3, "⚠️", "Missing Cells", f"{total_missing:,}",  f"{missing_pct}% of dataset",
        color="#F59E0B" if total_missing else "var(--green)")
    kpi(k4, "🔁", "Duplicates",    f"{duplicate_rows}",   f"{dup_pct}% of rows",
        color="#EF4444" if duplicate_rows else "var(--green)")
    kpi(k5, "✅", "Quality Score", f"{score}",            score_label, color=score_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column detail ────────────────────────────────────
    left, right = st.columns([1.2, 1])

    # Left: column-level breakdown
    with left:
        st.markdown("##### 📋 Column-Level Analysis")
        col_data = []
        for col in df.columns:
            miss     = int(df[col].isnull().sum())
            miss_p   = round(miss / total_rows * 100, 1) if total_rows else 0
            dtype    = str(df[col].dtype)
            n_unique = int(df[col].nunique())

            if miss == 0:
                status = "✅"
            elif miss_p < 10:
                status = "🟡"
            elif miss_p < 30:
                status = "🟠"
            else:
                status = "🔴"

            col_data.append({
                "Status": status,
                "Column": col,
                "Type": dtype,
                "Missing": f"{miss} ({miss_p}%)",
                "Unique": f"{n_unique}",
            })

        summary_df = pd.DataFrame(col_data)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 40 + 35 * len(col_data)),
        )

    # Right: flags & missing heatmap
    with right:
        st.markdown("##### 🚨 Quality Flags")

        flags = []

        # Missing columns
        missing_cols = missing_per_col[missing_per_col > 0]
        if missing_cols.empty:
            flags.append(("var(--green)", "No missing values detected across all columns."))
        else:
            for col, cnt in missing_cols.items():
                pct = round(cnt / total_rows * 100, 1)
                color = "#F59E0B" if pct < 20 else "#EF4444"
                flags.append((color, f"<b>{col}</b>: {cnt} missing ({pct}%)"))

        # Duplicates
        if duplicate_rows == 0:
            flags.append(("var(--green)", "No duplicate rows found."))
        else:
            flags.append(("#EF4444", f"<b>{duplicate_rows}</b> duplicate rows ({dup_pct}%)"))

        # Constant columns
        const_cols = [c for c in df.columns if df[c].nunique() <= 1]
        for c in const_cols:
            flags.append(("#7C3AED", f"<b>{c}</b> has only one unique value (constant column)"))

        # High-cardinality categoricals
        for c in cat_cols:
            if df[c].nunique() > 0.9 * total_rows and total_rows > 20:
                flags.append(("#F59E0B", f"<b>{c}</b> is nearly unique ({df[c].nunique()} values) — possible ID column"))

        items_html = "".join(
            f'<div class="dq-card"><div class="dq-status-dot" style="background:{color}"></div>'
            f'<div><div class="dq-card-detail">{msg}</div></div></div>'
            for color, msg in flags[:12]
        )
        st.markdown(items_html, unsafe_allow_html=True)

        # Column type breakdown
        st.markdown("##### 📊 Column Types")
        type_cols = st.columns(3)
        with type_cols[0]:
            st.metric("Numeric", len(num_cols))
        with type_cols[1]:
            st.metric("Categorical", len(cat_cols))
        with type_cols[2]:
            st.metric("DateTime", len(date_cols))

    # ── Missing Value Heatmap (if any missing) ───────────────
    if total_missing > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🗺️ Missing Value Map — click to expand", expanded=False):
            miss_map = df.isnull().astype(int)
            if miss_map.shape[0] > 200:
                miss_map = miss_map.head(200)
                st.caption("Showing first 200 rows")

            import plotly.express as px
            fig = px.imshow(
                miss_map.T,
                color_continuous_scale=[[0, "#111827"], [1, "#EF4444"]],
                labels={"x": "Row Index", "y": "Column", "color": "Missing"},
                aspect="auto",
            )
            fig.update_layout(
                paper_bgcolor="#0D1220",
                plot_bgcolor="#0D1220",
                font_color="#94A3B8",
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=20, b=0),
                height=max(200, 30 * len(df.columns)),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
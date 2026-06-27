import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _num_cols(df): return df.select_dtypes(include=np.number).columns.tolist()
def _cat_cols(df): return df.select_dtypes(include=["object","category"]).columns.tolist()

THEME = dict(
    paper_bgcolor="#0D1220", plot_bgcolor="#0D1220",
    font=dict(color="#94A3B8", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#1E2D45", linecolor="#1E2D45"),
    yaxis=dict(gridcolor="#1E2D45", linecolor="#1E2D45"),
    hoverlabel=dict(bgcolor="#111827", bordercolor="#1E2D45", font_color="#F0F4FF"),
)


# ── Individual insight generators ─────────────────────────────────────────────

def _distribution_insights(df):
    nums = _num_cols(df)
    findings = []
    for col in nums:
        s = df[col].dropna()
        skew = s.skew()
        kurt = s.kurtosis()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
        skew_tag = "right-skewed 📈" if skew>0.5 else "left-skewed 📉" if skew<-0.5 else "symmetric ✅"
        findings.append({
            "column": col, "skew": round(skew,2), "kurtosis": round(kurt,2),
            "outliers": outliers, "skew_tag": skew_tag,
            "mean": round(s.mean(),2), "std": round(s.std(),2),
        })
    return findings


def _correlation_insights(df):
    nums = _num_cols(df)
    if len(nums) < 2:
        return []
    corr = df[nums].corr()
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    pairs = upper.stack().reset_index()
    pairs.columns = ["col_a","col_b","r"]
    pairs["abs_r"] = pairs["r"].abs()
    pairs = pairs.sort_values("abs_r", ascending=False)
    findings = []
    for _, row in pairs.head(6).iterrows():
        strength = "Strong 🔴" if row.abs_r>0.7 else "Moderate 🟡" if row.abs_r>0.4 else "Weak ⚪"
        direction = "positive ↗" if row.r>0 else "negative ↘"
        findings.append({
            "col_a": row.col_a, "col_b": row.col_b,
            "r": round(row.r,3), "strength": strength, "direction": direction
        })
    return findings


def _trend_insights(df):
    nums = _num_cols(df)
    findings = []
    for col in nums[:5]:
        s = df[col].dropna().reset_index(drop=True)
        if len(s) < 4:
            continue
        half = len(s)//2
        first_half_mean = s.iloc[:half].mean()
        second_half_mean = s.iloc[half:].mean()
        pct_change = ((second_half_mean - first_half_mean) / abs(first_half_mean)*100
                      if first_half_mean != 0 else 0)
        direction = "Increasing 📈" if pct_change > 2 else "Decreasing 📉" if pct_change < -2 else "Stable ➡️"
        findings.append({
            "column": col, "pct_change": round(pct_change,1),
            "direction": direction,
            "first_mean": round(first_half_mean,2),
            "second_mean": round(second_half_mean,2),
        })
    return findings


def _categorical_insights(df):
    cats = _cat_cols(df)
    findings = []
    for col in cats[:4]:
        vc = df[col].value_counts()
        total = vc.sum()
        top_pct = round(vc.iloc[0]/total*100,1)
        concentration = "High 🔴" if top_pct>50 else "Medium 🟡" if top_pct>25 else "Low ✅"
        findings.append({
            "column": col, "n_unique": int(df[col].nunique()),
            "top_value": str(vc.index[0]), "top_pct": top_pct,
            "concentration": concentration,
            "top3_pct": round(vc.head(3).sum()/total*100,1),
        })
    return findings


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _dist_chart(df, col):
    s = df[col].dropna()
    fig = px.histogram(s, nbins=30, color_discrete_sequence=["#00D4FF"])
    fig.add_vline(x=s.mean(), line_dash="dash", line_color="#F59E0B",
                  annotation_text=f"Mean {s.mean():.1f}", annotation_font_color="#F59E0B")
    fig.add_vline(x=s.median(), line_dash="dot", line_color="#10B981",
                  annotation_text=f"Median {s.median():.1f}", annotation_font_color="#10B981")
    fig.update_traces(marker_line_color="#0D1220", marker_line_width=1)
    fig.update_layout(title=f"Distribution — {col}", **THEME)
    return fig


def _scatter_chart(df, col_a, col_b):
    fig = px.scatter(df, x=col_a, y=col_b, opacity=0.6,
                     color_discrete_sequence=["#7C3AED"])
    # Manual trendline using numpy (no statsmodels needed)
    import numpy as np
    clean = df[[col_a, col_b]].dropna()
    m, b = np.polyfit(clean[col_a], clean[col_b], 1)
    x_range = [clean[col_a].min(), clean[col_a].max()]
    y_range = [m*x_range[0]+b, m*x_range[1]+b]
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode="lines",
                             line=dict(color="#00D4FF", width=2, dash="dash"),
                             name="Trend"))
    fig.update_layout(title=f"{col_b} vs {col_a}", **THEME)
    return fig


def _trend_chart(df, col):
    s = df[col].dropna().reset_index(drop=True)
    fig = go.Figure(go.Scatter(
        y=s, mode="lines", line=dict(color="#00D4FF", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
    ))
    # Rolling average
    if len(s) >= 10:
        roll = s.rolling(max(3,len(s)//20)).mean()
        fig.add_trace(go.Scatter(
            y=roll, mode="lines", name="Trend",
            line=dict(color="#F59E0B", width=2, dash="dash"),
        ))
    fig.update_layout(title=f"{col} — Trend Over Records", showlegend=True, **THEME)
    return fig


def _cat_chart(df, col):
    vc = df[col].value_counts().head(15).reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(vc, x=col, y="count", color="count",
                 color_continuous_scale=["#1E2D45","#7C3AED"])
    fig.update_layout(title=f"{col} — Value Counts", coloraxis_showscale=False, **THEME)
    return fig


# ── Main component ────────────────────────────────────────────────────────────

def insights_panel(df: pd.DataFrame):
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
      <div class="section-label-dot"></div>
      <span class="section-label-text">Deep Analysis</span>
    </div>
    <div class="section-title">AI Insights Panel</div>
    <div class="section-desc">Automated pattern recognition across distributions, correlations, trends and categories.</div>
    """, unsafe_allow_html=True)

    nums = _num_cols(df)
    cats = _cat_cols(df)

    tab_labels = []
    if nums:   tab_labels += ["📊 Distributions", "📈 Trends"]
    if len(nums)>=2: tab_labels += ["🔗 Correlations"]
    if cats:   tab_labels += ["🏷️ Categories"]
    tab_labels += ["📋 Full Summary"]

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ── Tab: Distributions ────────────────────────────────────────────────────
    if nums:
        with tabs[tab_idx]:
            dist_findings = _distribution_insights(df)

            # Summary badges
            right_skewed = sum(1 for f in dist_findings if f["skew"]>0.5)
            left_skewed  = sum(1 for f in dist_findings if f["skew"]<-0.5)
            total_outliers = sum(f["outliers"] for f in dist_findings)

            b1,b2,b3 = st.columns(3)
            with b1:
                st.metric("Right-Skewed Columns", right_skewed)
            with b2:
                st.metric("Left-Skewed Columns", left_skewed)
            with b3:
                st.metric("Total Outliers Detected", total_outliers)

            st.markdown("<br>", unsafe_allow_html=True)

            col_select = st.selectbox("Select column to inspect", nums, key="dist_col")
            if col_select:
                left, right = st.columns([1.6, 1])
                with left:
                    st.plotly_chart(_dist_chart(df, col_select),
                                    use_container_width=True, config={"displayModeBar":False})
                with right:
                    f = next(x for x in dist_findings if x["column"]==col_select)
                    s = df[col_select].dropna()
                    st.markdown(f"""
                    <div class="insight-box" style="margin-top:0">
                      <div class="insight-box-title">📊 {col_select}</div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Shape: <b>{f['skew_tag']}</b> (skew={f['skew']})</div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Mean: <b>{f['mean']}</b> | Std Dev: <b>{f['std']}</b></div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Range: <b>{s.min():.2f}</b> → <b>{s.max():.2f}</b></div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Kurtosis: <b>{f['kurtosis']}</b> {'(heavy tails)' if f['kurtosis']>3 else '(normal tails)'}</div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Outliers: <b>{f['outliers']}</b> beyond 1.5×IQR</div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Missing: <b>{int(df[col_select].isnull().sum())}</b> values</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Mini box plot
                    fig_box = px.box(df, y=col_select, color_discrete_sequence=["#7C3AED"])
                    fig_box.update_layout(height=200, **THEME)
                    st.plotly_chart(fig_box, use_container_width=True,
                                    config={"displayModeBar":False}, key="box_dist")

        tab_idx += 1

    # ── Tab: Trends ───────────────────────────────────────────────────────────
    if nums:
        with tabs[tab_idx]:
            trend_findings = _trend_insights(df)

            if not trend_findings:
                st.info("Need more rows for trend analysis.")
            else:
                # Summary row
                increasing = sum(1 for f in trend_findings if "Increasing" in f["direction"])
                decreasing = sum(1 for f in trend_findings if "Decreasing" in f["direction"])
                stable     = sum(1 for f in trend_findings if "Stable"     in f["direction"])
                t1,t2,t3 = st.columns(3)
                with t1: st.metric("Increasing", increasing, delta="▲")
                with t2: st.metric("Decreasing", decreasing, delta="▼")
                with t3: st.metric("Stable", stable, delta="→")

                st.markdown("<br>", unsafe_allow_html=True)

                trend_col = st.selectbox("Select column", [f["column"] for f in trend_findings],
                                         key="trend_col")
                f = next(x for x in trend_findings if x["column"]==trend_col)

                st.plotly_chart(_trend_chart(df, trend_col),
                                use_container_width=True, config={"displayModeBar":False})

                pct = f["pct_change"]
                color = "#10B981" if pct>0 else "#EF4444" if pct<0 else "#94A3B8"
                st.markdown(f"""
                <div class="insight-box">
                  <div class="insight-box-title">📈 Trend Insight — {trend_col}</div>
                  <div class="insight-item"><span class="insight-bullet">▸</span>Direction: <b>{f['direction']}</b></div>
                  <div class="insight-item"><span class="insight-bullet">▸</span>First half avg: <b>{f['first_mean']}</b> → Second half avg: <b>{f['second_mean']}</b></div>
                  <div class="insight-item"><span class="insight-bullet">▸</span>Change: <b style="color:{color}">{pct:+.1f}%</b> across the dataset</div>
                </div>
                """, unsafe_allow_html=True)

        tab_idx += 1

    # ── Tab: Correlations ─────────────────────────────────────────────────────
    if len(nums) >= 2:
        with tabs[tab_idx]:
            corr_findings = _correlation_insights(df)

            # Heatmap
            corr_matrix = df[nums].corr().round(2)
            fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                 color_continuous_scale=[[0,"#7C3AED"],[0.5,"#111827"],[1,"#00D4FF"]],
                                 zmin=-1, zmax=1)
            fig_heat.update_layout(title="Full Correlation Matrix", **THEME)
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar":False})

            st.markdown("##### Top Correlated Pairs")
            pair_cols = st.columns(min(3, len(corr_findings)))
            for i, f in enumerate(corr_findings[:3]):
                with pair_cols[i]:
                    bar_color = "#10B981" if f["r"]>0 else "#EF4444"
                    st.markdown(f"""
                    <div class="kpi-card">
                      <div class="kpi-label">{f['col_a'][:14]} ↔ {f['col_b'][:14]}</div>
                      <div class="kpi-value" style="color:{bar_color}">{f['r']}</div>
                      <div class="kpi-sub">{f['strength']} · {f['direction']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Scatter explorer
            st.markdown("##### 🔍 Scatter Explorer")
            sc1, sc2 = st.columns(2)
            with sc1: x_col = st.selectbox("X axis", nums, key="sc_x")
            with sc2: y_col = st.selectbox("Y axis", nums, index=min(1,len(nums)-1), key="sc_y")
            if x_col != y_col:
                st.plotly_chart(_scatter_chart(df, x_col, y_col),
                                use_container_width=True, config={"displayModeBar":False})

        tab_idx += 1

    # ── Tab: Categories ───────────────────────────────────────────────────────
    if cats:
        with tabs[tab_idx]:
            cat_findings = _categorical_insights(df)

            cat_select = st.selectbox("Select categorical column", cats, key="cat_col")
            left, right = st.columns([1.6, 1])
            with left:
                st.plotly_chart(_cat_chart(df, cat_select),
                                use_container_width=True, config={"displayModeBar":False})
            with right:
                f = next((x for x in cat_findings if x["column"]==cat_select), None)
                if f:
                    conc_color = "#EF4444" if "High" in f["concentration"] else \
                                 "#F59E0B" if "Medium" in f["concentration"] else "#10B981"
                    st.markdown(f"""
                    <div class="insight-box" style="margin-top:0">
                      <div class="insight-box-title">🏷️ {cat_select}</div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Unique values: <b>{f['n_unique']}</b></div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Top value: <b>"{f['top_value']}"</b> ({f['top_pct']}%)</div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Top 3 cover: <b>{f['top3_pct']}%</b> of records</div>
                      <div class="insight-item"><span class="insight-bullet">▸</span>Concentration: <b style="color:{conc_color}">{f['concentration']}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Pie
                    vc = df[cat_select].value_counts().head(8).reset_index()
                    vc.columns = [cat_select, "count"]
                    fig_pie = px.pie(vc, names=cat_select, values="count", hole=0.4,
                                     color_discrete_sequence=["#00D4FF","#7C3AED","#10B981",
                                                               "#F59E0B","#EF4444","#F97316",
                                                               "#06B6D4","#8B5CF6"])
                    fig_pie.update_traces(textinfo="percent", marker=dict(line=dict(color="#0D1220",width=2)))
                    fig_pie.update_layout(height=220, showlegend=False, **THEME)
                    st.plotly_chart(fig_pie, use_container_width=True,
                                    config={"displayModeBar":False}, key="pie_cat")

        tab_idx += 1

    # ── Tab: Full Summary ─────────────────────────────────────────────────────
    with tabs[tab_idx]:
        st.markdown("##### 🧠 Complete Insight Report")

        dist_findings = _distribution_insights(df)
        trend_findings = _trend_insights(df)
        corr_findings  = _correlation_insights(df)
        cat_findings   = _categorical_insights(df)

        all_points = []

        for f in dist_findings:
            if f["outliers"] > 0:
                all_points.append(("⚠️", f"**{f['column']}** has {f['outliers']} outliers and is {f['skew_tag']}."))
            else:
                all_points.append(("✅", f"**{f['column']}** is {f['skew_tag']} with no outliers detected."))

        for f in trend_findings:
            all_points.append(("📈", f"**{f['column']}** is {f['direction']} ({f['pct_change']:+.1f}% change)."))

        for f in corr_findings[:3]:
            all_points.append(("🔗", f"**{f['col_a']}** ↔ **{f['col_b']}**: {f['strength']} {f['direction']} (r={f['r']})."))

        for f in cat_findings:
            all_points.append(("🏷️", f"**{f['column']}**: top value \"{f['top_value']}\" = {f['top_pct']}% ({f['concentration']} concentration)."))

        items = "".join(
            f'<div class="insight-item"><span class="insight-bullet">{ico}</span>{pt}</div>'
            for ico, pt in all_points
        )
        st.markdown(f'<div class="insight-box"><div class="insight-box-title">📋 All Findings</div>{items}</div>',
                    unsafe_allow_html=True)

        # Store findings for decision.py to use
        st.session_state["_insight_summary"] = all_points
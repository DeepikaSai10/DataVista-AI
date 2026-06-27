import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ── Plotly dark theme ─────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="#0D1220",
    plot_bgcolor="#0D1220",
    font=dict(color="#CBD5E1", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1E2D45", linecolor="#2A4066", zerolinecolor="#1E2D45",
               tickfont=dict(color="#CBD5E1"), title_font=dict(color="#F0F4FF")),
    yaxis=dict(gridcolor="#1E2D45", linecolor="#2A4066", zerolinecolor="#1E2D45",
               tickfont=dict(color="#CBD5E1"), title_font=dict(color="#F0F4FF")),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=["#00D4FF","#7C3AED","#10B981","#F59E0B","#EF4444",
              "#F97316","#06B6D4","#8B5CF6","#34D399","#FCD34D"],
    legend=dict(bgcolor="#111827", bordercolor="#2A4066", borderwidth=1,
                font=dict(color="#CBD5E1")),
    hoverlabel=dict(bgcolor="#111827", bordercolor="#2A4066", font_color="#F0F4FF"),
)

def _apply_theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F0F4FF")),
        **PLOTLY_THEME
    )
    return fig

def _num_cols(df): return df.select_dtypes(include=np.number).columns.tolist()
def _cat_cols(df): return df.select_dtypes(include=["object","category"]).columns.tolist()


# ── Smart Recommender ─────────────────────────────────────────────────────────

def recommend_charts(df: pd.DataFrame) -> list:
    """
    Analyze the dataset and return ranked chart recommendations.
    Each item: {chart, reason, score, cols}
    """
    nums = _num_cols(df)
    cats = _cat_cols(df)
    recs = []

    # Bar — best when categorical + numeric exist
    if cats and nums:
        top_cat = min(cats, key=lambda c: df[c].nunique())  # lowest cardinality cat
        recs.append({
            "chart": "Bar", "icon": "📊",
            "reason": f"Compare average <b>{nums[0]}</b> across <b>{top_cat}</b> groups ({df[top_cat].nunique()} categories).",
            "score": 90, "x": top_cat, "y": nums[0],
        })

    # Correlation Heatmap — when many numerics
    if len(nums) >= 3:
        recs.append({
            "chart": "Heatmap", "icon": "🔥",
            "reason": f"Reveal relationships between all <b>{len(nums)}</b> numeric columns at once.",
            "score": 88, "x": None, "y": None,
        })

    # Scatter — when 2+ numerics have moderate correlation
    if len(nums) >= 2:
        best_r, best_pair = 0, (nums[0], nums[1])
        for i in range(min(len(nums),5)):
            for j in range(i+1, min(len(nums),5)):
                r = abs(df[nums[i]].corr(df[nums[j]]))
                if r > best_r:
                    best_r, best_pair = r, (nums[i], nums[j])
        recs.append({
            "chart": "Scatter", "icon": "📍",
            "reason": f"Explore relationship between <b>{best_pair[0]}</b> and <b>{best_pair[1]}</b> (r={best_r:.2f}).",
            "score": 85, "x": best_pair[0], "y": best_pair[1],
        })

    # Pie — when a low-cardinality categorical exists
    if cats:
        pie_col = min(cats, key=lambda c: df[c].nunique())
        if df[pie_col].nunique() <= 10:
            recs.append({
                "chart": "Pie", "icon": "🥧",
                "reason": f"Show share breakdown of <b>{pie_col}</b> ({df[pie_col].nunique()} values).",
                "score": 80, "x": pie_col, "y": nums[0] if nums else None,
            })

    # Histogram — always useful for first numeric
    if nums:
        skew = abs(df[nums[0]].skew())
        recs.append({
            "chart": "Histogram", "icon": "📉",
            "reason": f"Understand distribution of <b>{nums[0]}</b> — {'skewed' if skew>0.5 else 'symmetric'} (skew={skew:.2f}).",
            "score": 78, "x": nums[0], "y": None,
        })

    # Box — when categorical + numeric with possible outliers
    if cats and nums:
        q1,q3 = df[nums[0]].quantile(0.25), df[nums[0]].quantile(0.75)
        iqr = q3-q1
        outliers = int(((df[nums[0]]<q1-1.5*iqr)|(df[nums[0]]>q3+1.5*iqr)).sum())
        if outliers > 0:
            recs.append({
                "chart": "Box", "icon": "📦",
                "reason": f"Detect <b>{outliers} outliers</b> in <b>{nums[0]}</b> split by <b>{cats[0]}</b>.",
                "score": 75, "x": cats[0], "y": nums[0],
            })

    # Treemap — when categorical + numeric
    if cats and nums:
        recs.append({
            "chart": "Treemap", "icon": "🌳",
            "reason": f"Visualize proportional size of <b>{cats[0]}</b> by total <b>{nums[0]}</b>.",
            "score": 72, "x": cats[0], "y": nums[0],
        })

    # Line — if enough rows
    if nums and len(df) >= 20:
        recs.append({
            "chart": "Line", "icon": "📈",
            "reason": f"Track how <b>{nums[0]}</b> changes across <b>{len(df):,}</b> records.",
            "score": 70, "x": None, "y": nums[0],
        })

    # Sort by score, return top 5
    return sorted(recs, key=lambda x: x["score"], reverse=True)[:5]


# ── Chart builders (accept x_col, y_col params) ───────────────────────────────

def build_bar(df, x_col, y_col):
    agg = df.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col, ascending=False).head(20)
    fig = px.bar(agg, x=x_col, y=y_col, color=y_col,
                 color_continuous_scale=["#1E2D45","#00D4FF"])
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color="#CBD5E1"),
                                       title=dict(font=dict(color="#F0F4FF"))))
    return _apply_theme(fig, f"Average {y_col} by {x_col}")

def build_line(df, x_col, y_col):
    fig = px.line(df.reset_index(), x=x_col or "index", y=y_col, line_shape="spline")
    fig.update_traces(line=dict(color="#00D4FF", width=2),
                      fill="tozeroy", fillcolor="rgba(0,212,255,0.07)")
    return _apply_theme(fig, f"{y_col} Trend")

def build_pie(df, x_col, y_col):
    if y_col:
        agg = df.groupby(x_col)[y_col].sum().reset_index().head(12)
        fig = px.pie(agg, names=x_col, values=y_col)
    else:
        vc = df[x_col].value_counts().head(12).reset_index()
        vc.columns = [x_col,"count"]
        fig = px.pie(vc, names=x_col, values="count")
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      hole=0.35, textfont=dict(color="#F0F4FF"),
                      marker=dict(line=dict(color="#0D1220", width=2)))
    return _apply_theme(fig, f"Distribution of {x_col}")

def build_histogram(df, x_col, y_col=None):
    fig = px.histogram(df, x=x_col, nbins=30, color_discrete_sequence=["#00D4FF"])
    fig.update_traces(marker_line_color="#0D1220", marker_line_width=1)
    return _apply_theme(fig, f"Distribution of {x_col}")

def build_box(df, x_col, y_col):
    if x_col and df[x_col].nunique() <= 20:
        fig = px.box(df, x=x_col, y=y_col, color=x_col,
                     color_discrete_sequence=PLOTLY_THEME["colorway"])
    else:
        fig = px.box(df, y=y_col, color_discrete_sequence=["#7C3AED"])
    return _apply_theme(fig, f"{y_col} Box Plot")

def build_heatmap(df, x_col=None, y_col=None):
    nums = _num_cols(df)
    corr = df[nums].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale=[[0,"#7C3AED"],[0.5,"#111827"],[1,"#00D4FF"]],
                    zmin=-1, zmax=1)
    fig.update_traces(textfont=dict(size=11, color="#F0F4FF"))
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color="#CBD5E1")))
    return _apply_theme(fig, "Correlation Heatmap")

def build_scatter(df, x_col, y_col):
    cats = _cat_cols(df)
    color_col = cats[0] if cats and df[cats[0]].nunique() <= 15 else None
    nums = _num_cols(df)
    size_col  = next((c for c in nums if c not in [x_col,y_col]), None)
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                     opacity=0.75, color_discrete_sequence=PLOTLY_THEME["colorway"])
    return _apply_theme(fig, f"{y_col} vs {x_col}")

def build_treemap(df, x_col, y_col):
    agg = df.groupby(x_col)[y_col].sum().reset_index().head(30)
    agg = agg[agg[y_col] > 0]
    fig = px.treemap(agg, path=[x_col], values=y_col, color=y_col,
                     color_continuous_scale=["#1E2D45","#00D4FF"])
    fig.update_traces(textfont=dict(color="#F0F4FF"))
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color="#CBD5E1")))
    return _apply_theme(fig, f"{y_col} Treemap by {x_col}")

def build_bubble(df, x_col, y_col):
    nums = _num_cols(df)
    cats = _cat_cols(df)
    size_col  = next((c for c in nums if c not in [x_col,y_col]), None)
    color_col = cats[0] if cats and df[cats[0]].nunique()<=15 else None
    if size_col:
        fig = px.scatter(df.head(300), x=x_col, y=y_col, size=size_col,
                         color=color_col, opacity=0.75, size_max=50,
                         color_discrete_sequence=PLOTLY_THEME["colorway"])
    else:
        fig = px.scatter(df.head(300), x=x_col, y=y_col, color=color_col,
                         opacity=0.75, color_discrete_sequence=PLOTLY_THEME["colorway"])
    return _apply_theme(fig, f"Bubble: {y_col} vs {x_col}")

def build_area(df, x_col, y_col):
    nums = _num_cols(df)
    cols_to_plot = [y_col] + [c for c in nums if c != y_col][:3]
    colors = ["#00D4FF","#7C3AED","#10B981","#F59E0B"]
    fig = go.Figure()
    for i, col in enumerate(cols_to_plot):
        r,g,b = int(colors[i%4][1:3],16), int(colors[i%4][3:5],16), int(colors[i%4][5:7],16)
        fig.add_trace(go.Scatter(
            y=df[col], x=df[x_col] if x_col else list(range(len(df))),
            name=col, mode="lines",
            line=dict(color=colors[i%4], width=1.5),
            stackgroup="one",
            fillcolor=f"rgba({r},{g},{b},0.15)",
        ))
    return _apply_theme(fig, "Stacked Area Chart")


# ── Column selector UI ────────────────────────────────────────────────────────

CHART_CONFIG = {
    "Bar":       {"needs": ["cat","num"],  "x_type": "cat", "y_type": "num",  "fn": build_bar},
    "Line":      {"needs": ["num"],        "x_type": "any", "y_type": "num",  "fn": build_line},
    "Pie":       {"needs": ["cat"],        "x_type": "cat", "y_type": "num?", "fn": build_pie},
    "Histogram": {"needs": ["num"],        "x_type": "num", "y_type": None,   "fn": build_histogram},
    "Box":       {"needs": ["num"],        "x_type": "cat?","y_type": "num",  "fn": build_box},
    "Heatmap":   {"needs": ["num","num"],  "x_type": None,  "y_type": None,   "fn": build_heatmap},
    "Scatter":   {"needs": ["num","num"],  "x_type": "num", "y_type": "num",  "fn": build_scatter},
    "Treemap":   {"needs": ["cat","num"],  "x_type": "cat", "y_type": "num",  "fn": build_treemap},
    "Bubble":    {"needs": ["num","num"],  "x_type": "num", "y_type": "num",  "fn": build_bubble},
    "Area":      {"needs": ["num"],        "x_type": "any", "y_type": "num",  "fn": build_area},
}

CHARTS = [
    ("📊","Bar"),("📈","Line"),("🥧","Pie"),("📉","Histogram"),("📦","Box"),
    ("🔥","Heatmap"),("📍","Scatter"),("🌳","Treemap"),("☁️","Bubble"),("📋","Area"),
]


def _col_selectors(df, chart_name: str):
    """Render X / Y column dropdowns based on chart type. Returns (x_col, y_col)."""
    nums = _num_cols(df)
    cats = _cat_cols(df)
    all_cols = df.columns.tolist()
    cfg = CHART_CONFIG.get(chart_name, {})
    x_type = cfg.get("x_type")
    y_type = cfg.get("y_type")

    x_col, y_col = None, None

    if x_type is None and y_type is None:
        # Heatmap — no selectors needed
        return None, None

    col1, col2 = st.columns(2)

    if x_type == "cat":
        opts = cats if cats else all_cols
        with col1:
            x_col = st.selectbox("X axis (Category)", opts, key=f"x_{chart_name}")
    elif x_type == "num":
        opts = nums if nums else all_cols
        with col1:
            x_col = st.selectbox("X axis (Numeric)", opts, key=f"x_{chart_name}")
    elif x_type == "cat?":
        opts = ["— None —"] + cats
        with col1:
            sel = st.selectbox("X axis (Category, optional)", opts, key=f"x_{chart_name}")
            x_col = None if sel == "— None —" else sel
    elif x_type == "any":
        opts = ["— Index —"] + all_cols
        with col1:
            sel = st.selectbox("X axis", opts, key=f"x_{chart_name}")
            x_col = None if sel == "— Index —" else sel

    if y_type == "num":
        opts = nums if nums else all_cols
        with col2:
            y_col = st.selectbox("Y axis (Numeric)", opts, key=f"y_{chart_name}")
    elif y_type == "num?":
        opts = ["— Count —"] + nums
        with col2:
            sel = st.selectbox("Y axis (Numeric, optional)", opts, key=f"y_{chart_name}")
            y_col = None if sel == "— Count —" else sel

    return x_col, y_col


# ── Insight generator ─────────────────────────────────────────────────────────

def generate_insight(chart_type, df, x_col=None, y_col=None):
    nums = _num_cols(df); cats = _cat_cols(df)
    insights = []

    try:
        if chart_type == "Bar" and x_col and y_col:
            agg = df.groupby(x_col)[y_col].mean()
            top, bot = agg.idxmax(), agg.idxmin()
            insights.append(f"**{top}** has the highest avg {y_col} ({agg[top]:.2f}); **{bot}** the lowest ({agg[bot]:.2f}).")
            spread = agg.max() - agg.min()
            insights.append(f"Range across groups: **{spread:.2f}** — {'high variation' if spread > agg.mean() else 'fairly uniform'}.")
            insights.append(f"Top 3 groups = **{round(agg.nlargest(3).sum()/agg.sum()*100,1)}%** of total.")

        elif chart_type == "Line" and y_col:
            s = df[y_col].dropna()
            trend = "upward 📈" if s.iloc[-1] > s.iloc[0] else "downward 📉"
            insights.append(f"**{y_col}** shows a **{trend}** trend across records.")
            insights.append(f"Peak: **{s.max():.2f}** at index {s.idxmax()} | Low: **{s.min():.2f}**.")
            vol = s.std()/s.mean()*100 if s.mean()!=0 else 0
            insights.append(f"Volatility (CV): **{vol:.1f}%** — {'unstable' if vol>30 else 'stable'}.")

        elif chart_type == "Pie" and x_col:
            vc = df[x_col].value_counts()
            insights.append(f"**{vc.index[0]}** is the largest slice at **{round(vc.iloc[0]/vc.sum()*100,1)}%**.")
            insights.append(f"Top 3 values cover **{round(vc.head(3).sum()/vc.sum()*100,1)}%** of all records.")
            insights.append(f"**{df[x_col].nunique()}** unique categories in **{x_col}**.")

        elif chart_type == "Histogram" and x_col:
            s = df[x_col].dropna()
            skew = s.skew()
            insights.append(f"Distribution is **{'right-skewed' if skew>0.5 else 'left-skewed' if skew<-0.5 else 'symmetric'}** (skew={skew:.2f}).")
            insights.append(f"Mean **{s.mean():.2f}** vs median **{s.median():.2f}**.")
            q1,q3=s.quantile(0.25),s.quantile(0.75); iqr=q3-q1
            out = int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
            if out: insights.append(f"**{out}** potential outlier(s) beyond 1.5×IQR.")

        elif chart_type == "Box" and y_col:
            s = df[y_col].dropna()
            q1,q3=s.quantile(0.25),s.quantile(0.75); iqr=q3-q1
            out = int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
            insights.append(f"IQR of **{y_col}**: **{iqr:.2f}** (Q1={q1:.2f}, Q3={q3:.2f}).")
            insights.append(f"**{out}** outlier(s) detected." if out else "No outliers beyond whiskers.")
            if x_col and df[x_col].nunique()<=20:
                spread = df.groupby(x_col)[y_col].std()
                insights.append(f"**{spread.idxmax()}** has the most within-group variance.")

        elif chart_type == "Heatmap":
            corr = df[nums].corr()
            upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
            s = upper.stack()
            mp = s.abs().idxmax(); mv = s[mp]
            insights.append(f"Strongest: **{mp[0]}** ↔ **{mp[1]}** (r={mv:.2f}).")
            insights.append(f"**{(s.abs()>0.7).sum()}** pair(s) with |r|>0.7 — watch for multicollinearity.")

        elif chart_type == "Scatter" and x_col and y_col:
            r = df[[x_col,y_col]].dropna().corr().iloc[0,1]
            strength = "strong" if abs(r)>0.7 else "moderate" if abs(r)>0.4 else "weak"
            insights.append(f"**{strength} {'positive' if r>0 else 'negative'}** correlation (r={r:.2f}).")
            insights.append(f"As **{x_col}** increases, **{y_col}** tends to {'increase ↗' if r>0 else 'decrease ↘'}.")

        elif chart_type == "Treemap" and x_col and y_col:
            agg = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
            insights.append(f"**{agg.index[0]}** is the largest segment ({agg.iloc[0]:.2f}).")
            insights.append(f"Top 3 = **{round(agg.head(3).sum()/agg.sum()*100,1)}%** of total {y_col}.")

    except Exception:
        pass

    if not insights:
        insights = [
            f"Dataset: **{len(df):,}** rows × **{len(df.columns)}** columns.",
            f"**{len(nums)}** numeric, **{len(cats)}** categorical columns available.",
            "Adjust column selections above for more specific insights.",
        ]
    return insights


# ── Main gallery ──────────────────────────────────────────────────────────────

def visualization_gallery(df: pd.DataFrame):
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
      <div class="section-label-dot"></div>
      <span class="section-label-text">Visualization Engine</span>
    </div>
    <div class="section-title">Visualization Gallery</div>
    <div class="section-desc">
      Pick a recommended chart below, or choose any chart type manually.
    </div>
    """, unsafe_allow_html=True)

    if "active_chart"   not in st.session_state: st.session_state.active_chart   = None
    if "chart_history"  not in st.session_state: st.session_state.chart_history  = []
    if "chart_x"        not in st.session_state: st.session_state.chart_x        = {}
    if "chart_y"        not in st.session_state: st.session_state.chart_y        = {}

    # ── Smart Recommendations ─────────────────────────────────────────────────
    recs = recommend_charts(df)

    st.markdown("""
    <div style="color:#F0F4FF;font-size:1rem;font-weight:600;margin-bottom:0.75rem">
      ✨ Smart Recommendations — based on your data
    </div>
    """, unsafe_allow_html=True)

    rec_cols = st.columns(len(recs))
    for i, rec in enumerate(recs):
        with rec_cols[i]:
            is_active = st.session_state.active_chart == rec["chart"]
            border = "2px solid #00D4FF" if is_active else "1px solid #2A4066"
            bg     = "rgba(0,212,255,0.08)" if is_active else "#111827"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:12px;
                        padding:0.9rem 0.75rem;text-align:center;min-height:110px">
              <div style="font-size:1.6rem">{rec['icon']}</div>
              <div style="font-size:12px;font-weight:700;color:#F0F4FF;margin:4px 0 6px">{rec['chart']}</div>
              <div style="font-size:10px;color:#94A3B8;line-height:1.4">{rec['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Use {rec['chart']}", key=f"rec_{rec['chart']}_{i}",
                         use_container_width=True):
                st.session_state.active_chart = rec["chart"]
                # Pre-fill columns from recommendation
                if rec.get("x"): st.session_state.chart_x[rec["chart"]] = rec["x"]
                if rec.get("y"): st.session_state.chart_y[rec["chart"]] = rec["y"]
                if rec["chart"] not in st.session_state.chart_history:
                    st.session_state.chart_history.append(rec["chart"])
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── All chart icons ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="color:#94A3B8;font-size:0.85rem;font-weight:500;margin-bottom:0.6rem;
                letter-spacing:1px;text-transform:uppercase">
      Or choose manually
    </div>
    """, unsafe_allow_html=True)

    icon_cols = st.columns(len(CHARTS))
    for i, (icon, name) in enumerate(CHARTS):
        is_active = st.session_state.active_chart == name
        with icon_cols[i]:
            border = "2px solid #00D4FF" if is_active else "1px solid #1E2D45"
            bg     = "rgba(0,212,255,0.08)" if is_active else "#111827"
            txt_c  = "#00D4FF" if is_active else "#94A3B8"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:12px;
                        padding:0.7rem 0.4rem;text-align:center">
              <div style="font-size:1.6rem">{icon}</div>
              <div style="font-size:10px;font-weight:600;color:{txt_c};margin-top:3px">{name}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("▶", key=f"chart_btn_{name}", use_container_width=True,
                         help=f"Select {name}"):
                st.session_state.active_chart = name
                if name not in st.session_state.chart_history:
                    st.session_state.chart_history.append(name)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Active chart with column selectors ────────────────────────────────────
    if st.session_state.active_chart:
        _render_chart_block(df, st.session_state.active_chart)

    # ── History ───────────────────────────────────────────────────────────────
    history = [c for c in st.session_state.chart_history
               if c != st.session_state.active_chart]
    if history:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#94A3B8;font-size:0.8rem;font-weight:600;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem">
          🕘 Previously Viewed
        </div>
        """, unsafe_allow_html=True)
        hcols = st.columns(min(len(history), 5))
        for i, name in enumerate(reversed(history[-5:])):
            icon = next(ic for ic, n in CHARTS if n == name)
            with hcols[i]:
                if st.button(f"{icon} {name}", key=f"hist_{name}", use_container_width=True):
                    st.session_state.active_chart = name
                    st.rerun()


def _render_chart_block(df, chart_name):
    cfg = CHART_CONFIG.get(chart_name)
    if not cfg:
        return

    icon = next(ic for ic, n in CHARTS if n == chart_name)

    st.markdown(f"""
    <div style="background:#111827;border:1px solid #2A4066;border-radius:16px;
                padding:1.5rem 1.5rem 0.5rem;margin-bottom:1rem">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem">
        <span style="font-size:1.5rem">{icon}</span>
        <span style="font-family:Space Grotesk,sans-serif;font-size:1.1rem;
                     font-weight:700;color:#F0F4FF">{chart_name} Chart</span>
      </div>
    """, unsafe_allow_html=True)

    # Column selectors inside the card
    x_col, y_col = _col_selectors(df, chart_name)

    st.markdown("</div>", unsafe_allow_html=True)

    # Build chart
    fn = cfg["fn"]
    try:
        if cfg["x_type"] is None and cfg["y_type"] is None:
            fig = fn(df, None, None)
        elif cfg["y_type"] is None:
            fig = fn(df, x_col)
        else:
            fig = fn(df, x_col, y_col)
    except Exception as e:
        st.warning(f"⚠️ Could not render **{chart_name}**: {e}")
        return

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # Insight panel
    insights = generate_insight(chart_name, df, x_col, y_col)
    items_html = "".join(
        f'<div class="insight-item"><span class="insight-bullet">▸</span>{pt}</div>'
        for pt in insights
    )
    st.markdown(f"""
    <div class="insight-box">
      <div class="insight-box-title">🧠 AI Insights — {chart_name}</div>
      {items_html}
    </div>
    """, unsafe_allow_html=True)
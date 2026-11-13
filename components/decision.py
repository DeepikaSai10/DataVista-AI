import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


from components.utils import get_num_cols, get_cat_cols

def _num_cols(df): return get_num_cols(df)
def _cat_cols(df): return get_cat_cols(df)

THEME = dict(
    paper_bgcolor="#0D1220", plot_bgcolor="#0D1220",
    font=dict(color="#94A3B8", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=dict(bgcolor="#111827", bordercolor="#1E2D45", font_color="#F0F4FF"),
)


# ── Decision engine ───────────────────────────────────────────────────────────

def _generate_decisions(df: pd.DataFrame) -> dict:
    nums = _num_cols(df)
    cats = _cat_cols(df)
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    missing_pct = df.isnull().sum().sum() / total_cells * 100
    dup_count   = int(df.duplicated().sum())

    actions      = []   # immediate actions
    watch        = []   # watch list
    opportunities= []   # growth opportunities
    risks        = []   # risk flags

    # ── Data quality actions ──────────────────────────────────────────────────
    if missing_pct > 20:
        actions.append({
            "priority": "HIGH", "icon": "🔴",
            "title": "Critical: High Missing Data",
            "body": f"{missing_pct:.1f}% of cells are empty. Consider imputation strategies "
                    f"(median fill for numerics, mode fill for categoricals) or dropping "
                    f"columns with >50% missing before modeling.",
            "tags": ["Data Cleaning", "Imputation"],
        })
    elif missing_pct > 5:
        watch.append({
            "priority": "MEDIUM", "icon": "🟡",
            "title": "Missing Values Need Attention",
            "body": f"{missing_pct:.1f}% missing. Review columns with highest gaps before "
                    f"running statistical tests — missing data can bias results.",
            "tags": ["Data Quality"],
        })
    else:
        opportunities.append({
            "priority": "INFO", "icon": "✅",
            "title": "Clean Dataset — Ready for Analysis",
            "body": f"Only {missing_pct:.1f}% missing. Dataset is well-formed and suitable "
                    f"for direct modeling or reporting.",
            "tags": ["Ready", "Modeling"],
        })

    if dup_count > 0:
        dup_pct = round(dup_count/total_rows*100,1)
        (actions if dup_pct>5 else watch).append({
            "priority": "HIGH" if dup_pct>5 else "MEDIUM",
            "icon": "🔴" if dup_pct>5 else "🟡",
            "title": f"Duplicate Rows Detected ({dup_count})",
            "body": f"{dup_pct}% of rows are exact duplicates. "
                    f"Remove with `df.drop_duplicates()` before any aggregation or modeling.",
            "tags": ["Deduplication", "Data Cleaning"],
        })

    # ── Numeric insights → decisions ─────────────────────────────────────────
    if nums:
        # Outlier recommendation
        total_outliers = 0
        flagged_cols = []
        for col in nums:
            s = df[col].dropna()
            q1,q3 = s.quantile(0.25), s.quantile(0.75); iqr=q3-q1
            n = int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
            total_outliers += n
            if n > 0:
                flagged_cols.append(f"{col} ({n})")

        if total_outliers > 0:
            watch.append({
                "priority": "MEDIUM", "icon": "📊",
                "title": f"Outliers in {len(flagged_cols)} Column(s)",
                "body": f"Columns: {', '.join(flagged_cols[:4])}. "
                        f"Investigate whether these are data errors or valid extremes. "
                        f"Consider capping/winsorizing if they distort your model.",
                "tags": ["Outliers", "Feature Engineering"],
            })

        # Skewness → transformation
        high_skew = [(c, round(df[c].skew(),2)) for c in nums if abs(df[c].skew())>1]
        if high_skew:
            names = ", ".join(f"{c} ({s})" for c,s in high_skew[:4])
            opportunities.append({
                "priority": "INFO", "icon": "📐",
                "title": "Skewed Columns — Log Transform Recommended",
                "body": f"{names}. Apply log or Box-Cox transformation to normalize "
                        f"these distributions before regression or clustering.",
                "tags": ["Feature Engineering", "Transformation"],
            })

        # Correlation → multicollinearity
        if len(nums) >= 2:
            corr = df[nums].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
            high_corr_pairs = [(i,j,round(upper.loc[i,j],2))
                               for i in upper.index for j in upper.columns
                               if pd.notna(upper.loc[i,j]) and upper.loc[i,j]>0.85]
            if high_corr_pairs:
                pair_str = "; ".join(f"{a}↔{b} ({r})" for a,b,r in high_corr_pairs[:3])
                risks.append({
                    "priority": "HIGH", "icon": "⚡",
                    "title": "Multicollinearity Risk Detected",
                    "body": f"Highly correlated pairs: {pair_str}. "
                            f"Drop or combine one from each pair to avoid "
                            f"inflated variance in regression models.",
                    "tags": ["Multicollinearity", "Modeling Risk"],
                })

        # Size → recommendation
        if total_rows < 100:
            risks.append({
                "priority": "MEDIUM", "icon": "📏",
                "title": "Small Dataset — Limited Statistical Power",
                "body": f"Only {total_rows} rows. Statistical significance thresholds "
                        f"are harder to achieve. Collect more data or use bootstrap methods.",
                "tags": ["Sample Size", "Statistics"],
            })
        elif total_rows > 100_000:
            opportunities.append({
                "priority": "INFO", "icon": "🚀",
                "title": "Large Dataset — Ideal for Machine Learning",
                "body": f"{total_rows:,} rows enables deep learning and ensemble methods. "
                        f"Consider gradient boosting (XGBoost, LightGBM) or neural networks.",
                "tags": ["ML Opportunity", "Big Data"],
            })

    # ── Categorical insights → decisions ─────────────────────────────────────
    if cats:
        high_card = [(c, df[c].nunique()) for c in cats if df[c].nunique()>50]
        if high_card:
            names = ", ".join(f"{c} ({n} values)" for c,n in high_card[:3])
            watch.append({
                "priority": "MEDIUM", "icon": "🏷️",
                "title": "High Cardinality Columns",
                "body": f"{names}. These may be ID fields or free text. "
                        f"Consider target encoding, frequency encoding, or dropping for modeling.",
                "tags": ["Encoding", "Feature Engineering"],
            })

        const_cats = [c for c in cats if df[c].nunique()<=1]
        if const_cats:
            actions.append({
                "priority": "HIGH", "icon": "🗑️",
                "title": "Constant Columns Detected",
                "body": f"Columns with a single value: {', '.join(const_cats)}. "
                        f"These carry zero predictive value — drop them.",
                "tags": ["Cleanup", "Feature Selection"],
            })

    # Always add at least one opportunity
    if not opportunities:
        opportunities.append({
            "priority": "INFO", "icon": "🧠",
            "title": "Explore Segmentation",
            "body": f"With {total_cols} features and {total_rows} rows, "
                    f"try clustering (K-Means, DBSCAN) to discover natural segments.",
            "tags": ["Clustering", "Segmentation"],
        })

    return {
        "actions": actions,
        "watch": watch,
        "opportunities": opportunities,
        "risks": risks,
    }


def _score_gauge(score: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x":[0,1],"y":[0,1]},
        title={"text":"Decision Readiness Score", "font":{"color":"#94A3B8","size":13}},
        gauge={
            "axis":{"range":[0,100], "tickcolor":"#1E2D45"},
            "bar":{"color":"#00D4FF"},
            "bgcolor":"#111827",
            "bordercolor":"#1E2D45",
            "steps":[
                {"range":[0,40],  "color":"rgba(239,68,68,0.15)"},
                {"range":[40,70], "color":"rgba(245,158,11,0.15)"},
                {"range":[70,100],"color":"rgba(16,185,129,0.15)"},
            ],
            "threshold":{"line":{"color":"#00D4FF","width":3},"thickness":0.8,"value":score},
        },
        number={"font":{"color":"#F0F4FF","size":36},"suffix":"/100"},
    ))
    fig.update_layout(height=220, paper_bgcolor="#0D1220",
                      font=dict(color="#94A3B8"), margin=dict(l=20,r=20,t=30,b=10))
    return fig


def _card(item: dict):
    color_map = {"HIGH":"#EF4444","MEDIUM":"#F59E0B","INFO":"#10B981","LOW":"#94A3B8"}
    c = color_map.get(item["priority"], "#94A3B8")
    tags_html = "".join(f'<span class="decision-chip">{t}</span>' for t in item["tags"])
    st.markdown(f"""
    <div class="decision-card" style="border-left:3px solid {c};padding-left:1.5rem">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem">
        <span style="font-size:1.4rem">{item['icon']}</span>
        <span class="decision-card-title">{item['title']}</span>
        <span style="margin-left:auto;background:{c}22;color:{c};border:1px solid {c}44;
                     border-radius:100px;padding:2px 10px;font-size:11px;font-weight:700">
          {item['priority']}
        </span>
      </div>
      <p style="color:#94A3B8;font-size:0.875rem;margin:0 0 0.75rem">{item['body']}</p>
      <div>{tags_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main component ────────────────────────────────────────────────────────────

def decision_support(df: pd.DataFrame):
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
      <div class="section-label-dot"></div>
      <span class="section-label-text">Strategic Intelligence</span>
    </div>
    <div class="section-title">Decision Support</div>
    <div class="section-desc">Auto-generated recommendations based on your data's patterns, quality, and structure.</div>
    """, unsafe_allow_html=True)

    decisions = _generate_decisions(df)

    n_actions = len(decisions["actions"])
    n_risks   = len(decisions["risks"])
    n_watch   = len(decisions["watch"])
    n_opps    = len(decisions["opportunities"])

    # Score = start at 100, deduct for issues
    score = 100
    score -= n_actions * 15
    score -= n_risks   * 10
    score -= n_watch   *  5
    score  = max(10, min(100, score))

    readiness = "High 🟢" if score>=70 else "Medium 🟡" if score>=40 else "Low 🔴"

    # ── Overview row ──────────────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.plotly_chart(_score_gauge(score), use_container_width=True,
                        config={"displayModeBar":False})
        ov1, ov2 = st.columns(2)
        with ov1:
            st.metric("Readiness", readiness)
            st.metric("Risks", n_risks, delta=f"-{n_risks}" if n_risks else None,
                      delta_color="inverse")
        with ov2:
            st.metric("Actions", n_actions, delta=f"-{n_actions}" if n_actions else None,
                      delta_color="inverse")
            st.metric("Opportunities", n_opps)

    with right:
        # Priority matrix mini chart
        cats_chart = ["Immediate Actions","Risks","Watch List","Opportunities"]
        vals_chart = [n_actions, n_risks, n_watch, n_opps]
        colors_c   = ["#EF4444","#F97316","#F59E0B","#10B981"]
        fig_bar = go.Figure(go.Bar(
            x=cats_chart, y=vals_chart,
            marker_color=colors_c,
            marker_line_color="#0D1220", marker_line_width=2,
            text=vals_chart, textposition="outside",
            textfont=dict(color="#F0F4FF"),
        ))
        fig_bar.update_layout(
            title="Recommendation Breakdown",
            yaxis=dict(gridcolor="#1E2D45", linecolor="#1E2D45", tickcolor="#1E2D45"),
            xaxis=dict(linecolor="#1E2D45"),
            **THEME
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs for each category ────────────────────────────────────────────────
    tab_labels = [
        f"🔴 Actions ({n_actions})",
        f"⚡ Risks ({n_risks})",
        f"🟡 Watch ({n_watch})",
        f"✅ Opportunities ({n_opps})",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if decisions["actions"]:
            for item in decisions["actions"]: _card(item)
        else:
            st.success("✅ No immediate actions required — dataset looks clean!")

    with tabs[1]:
        if decisions["risks"]:
            for item in decisions["risks"]: _card(item)
        else:
            st.success("✅ No significant risks detected.")

    with tabs[2]:
        if decisions["watch"]:
            for item in decisions["watch"]: _card(item)
        else:
            st.info("Nothing on the watch list.")

    with tabs[3]:
        if decisions["opportunities"]:
            for item in decisions["opportunities"]: _card(item)
        else:
            st.info("No specific opportunities identified yet.")

    # Store for report export
    st.session_state["_decisions"] = decisions
    st.session_state["_decision_score"] = score
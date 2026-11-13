import streamlit as st
import pandas as pd

from components.upload import upload_section
from components.data_quality import data_quality_report
from components.kpi import kpi_dashboard
from components.charts import visualization_gallery
from components.insights import insights_panel
from components.decision import decision_support
from components.utils import assess_data

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataVista AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}
/* Nuclear dark background override */
html, body { background-color: #070B14 !important; }
.stApp { background-color: #070B14 !important; }
[data-testid="stAppViewContainer"] { background-color: #070B14 !important; }
[data-testid="stHeader"] { background-color: #070B14 !important; }
.block-container { padding-top:0 !important; padding-left:3rem !important; padding-right:3rem !important; }
/* Hide blank button labels */
[data-testid="stBaseButton-secondary"] p { font-size:11px !important; color:#94A3B8 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">✦ AI-Powered Analytics</div>
  <h1>DataVista AI</h1>
  <div class="hero-sub">Transform Raw Data into Intelligent Business Decisions</div>
  <div class="hero-pills">
    <span class="hero-pill">📊 Visualize</span>
    <span class="hero-pill">🛡️ Quality Check</span>
    <span class="hero-pill">🧠 AI Insights</span>
    <span class="hero-pill">🚀 Decision Support</span>
    <span class="hero-pill">📥 Export Reports</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CARDS
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
features = [
    ("📊", "Interactive Dashboard", "10 chart types — bar, line, pie, scatter, heatmap, treemap and more."),
    ("🛡️", "Data Quality Engine",  "Missing value detection, duplicate checks, and per-column health scores."),
    ("🧠", "AI Insight Generator", "Automatic pattern recognition and data-driven observations per chart."),
    ("🚀", "Decision Support",     "Business-ready summaries from your data — no analyst required."),
]
for col, (icon, title, desc) in zip([c1,c2,c3,c4], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
          <h2>{icon}</h2>
          <h4>{title}</h4>
          <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-label" style="justify-content:center;margin-bottom:0.5rem">
  <div class="section-label-dot"></div>
  <span class="section-label-text">Step 1 — Import</span>
</div>
<div class="upload-title">Upload Your Dataset</div>
<p style="text-align:center;color:#94A3B8;font-size:1rem;margin-bottom:1.5rem">
  Supports CSV and Excel files — drag &amp; drop or click to browse.
</p>
""", unsafe_allow_html=True)

df = upload_section()

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if df is not None:
    st.success(f"✅ Dataset loaded — **{len(df):,} rows × {len(df.columns)} columns**")

    # Dataset preview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">
      <div class="section-label-dot"></div>
      <span class="section-label-text">Dataset Preview</span>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=350)

    # All sections
    # Health + KPIs always show, so users can see what is wrong
    data_quality_report(df)
    kpi_dashboard(df)

    # Gate: decide whether the data is safe to chart
    check = assess_data(df)

    if check["level"] == "blocked":
        issues_md = "\n".join(f"- {i}" for i in check["issues"][:6])
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.error(
            f"🛑 **Data quality is too low to analyze reliably (score {check['score']}/100).**\n\n"
            f"{issues_md}\n\n"
            "Please clean these issues and re-upload your file."
        )
        st.stop()

    if check["level"] == "warning":
        issues_md = "\n".join(f"- {i}" for i in check["issues"][:6])
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.warning(
            f"⚠️ **Your data has quality issues that can make charts misleading:**\n\n{issues_md}\n\n"
            "For the most reliable results, clean these and re-upload."
        )
        proceed_key = f"proceed_{abs(hash(tuple(df.columns)))}_{len(df)}"
        if not st.checkbox("I understand the risks, continue anyway", key=proceed_key):
            st.stop()

    # Charts, insights and decisions only run after the gate
    visualization_gallery(df)
    insights_panel(df)
    decision_support(df)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#1E2D45;font-size:12px;padding:2rem 0 1rem;
                border-top:1px solid #1E2D45;margin-top:2rem">
      DataVista AI · Built with Streamlit &amp; Plotly
    </div>
    """, unsafe_allow_html=True)
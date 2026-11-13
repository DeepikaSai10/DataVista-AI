"""
report.py  —  PDF Report Generator for DataVista AI
Generates a professional multi-section PDF using ReportLab.
"""

import io
import datetime
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.platypus import PageBreak, CondPageBreak
from reportlab.lib.colors import HexColor


# ── Brand colors ──────────────────────────────────────────────────────────────
BG       = HexColor("#070B14")
SURFACE  = HexColor("#0D1220")
CARD     = HexColor("#111827")
BORDER   = HexColor("#1E2D45")
CYAN     = HexColor("#00D4FF")
VIOLET   = HexColor("#7C3AED")
GREEN    = HexColor("#10B981")
AMBER    = HexColor("#F59E0B")
RED      = HexColor("#EF4444")
TEXT     = HexColor("#F0F4FF")
MUTED    = HexColor("#94A3B8")
WHITE    = HexColor("#FFFFFF")


from components.utils import get_num_cols, get_cat_cols

def _num_cols(df): return get_num_cols(df)
def _cat_cols(df): return get_cat_cols(df)
def _fmt(v):
    if abs(v)>=1_000_000: return f"{v/1_000_000:.2f}M"
    if abs(v)>=1_000:     return f"{v/1_000:.1f}K"
    return f"{v:.2f}"
def _w(widths, total=17 * cm):
    """Turn widths like '5cm' into numbers and shrink them to fit the page."""
    out = []
    for w in widths:
        if isinstance(w, str):
            w = w.strip().lower()
            if w.endswith("cm"):
                out.append(float(w[:-2]) * cm)
            elif w.endswith("%"):
                out.append(float(w[:-1]) / 100 * total)
            else:
                out.append(float(w))
        else:
            out.append(float(w))
    s = sum(out)
    if s > total:
        out = [x * total / s for x in out]
    return out

# ── Styles ────────────────────────────────────────────────────────────────────

def _make_styles():
    s = getSampleStyleSheet()

    def style(name, **kw):
        return ParagraphStyle(name, parent=s["Normal"], **kw)

    return {
        "title":    style("DV_Title",    fontSize=28, textColor=WHITE,
                          fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_LEFT, leading=34),
        "subtitle": style("DV_Sub",      fontSize=13, textColor=MUTED,
                          fontName="Helvetica",      spaceAfter=16),
        "h1":       style("DV_H1",       fontSize=16, textColor=CYAN,
                          fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8, leading=20),
        "h2":       style("DV_H2",       fontSize=12, textColor=WHITE,
                          fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=6, leading=15),
        "body":     style("DV_Body",     fontSize=9,  textColor=MUTED,
                          fontName="Helvetica",       spaceAfter=6, leading=14),
        "bullet":   style("DV_Bullet",   fontSize=9,  textColor=MUTED,
                          fontName="Helvetica",       spaceAfter=4, leftIndent=14,
                          bulletIndent=4, leading=13),
        "small":    style("DV_Small",    fontSize=8,  textColor=MUTED,
                          fontName="Helvetica",       spaceAfter=4),
        "badge_g":  style("DV_BadgeG",   fontSize=9,  textColor=GREEN,
                          fontName="Helvetica-Bold"),
        "badge_r":  style("DV_BadgeR",   fontSize=9,  textColor=RED,
                          fontName="Helvetica-Bold"),
        "badge_a":  style("DV_BadgeA",   fontSize=9,  textColor=AMBER,
                          fontName="Helvetica-Bold"),
        "meta":     style("DV_Meta",     fontSize=8,  textColor=MUTED,
                          fontName="Helvetica",       alignment=TA_RIGHT),
    }


# ── Table helpers ─────────────────────────────────────────────────────────────

def _header_table(df, styles):
    """Cover header band."""
    now = datetime.datetime.now().strftime("%B %d, %Y — %H:%M")
    data = [[
        Paragraph("DataVista AI", styles["title"]),
        Paragraph(f"Generated: {now}", styles["meta"]),
    ]]
    t = Table(data, colWidths=["62%","38%"])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), SURFACE),
        ("TOPPADDING",  (0,0),(-1,-1), 18),
        ("BOTTOMPADDING",(0,0),(-1,-1), 18),
        ("LEFTPADDING", (0,0),(-1,-1), 20),
        ("RIGHTPADDING",(0,0),(-1,-1), 20),
        ("ROUNDEDCORNERS", [6]),
        ("LINEBELOW", (0,0),(-1,-1), 2, CYAN),
    ]))
    return t


def _kpi_row(label_vals: list, styles):
    """A row of KPI boxes: [(label, value, color), ...]"""
    n = len(label_vals)
    size = 14 if n >= 6 else 18          # many boxes -> smaller numbers so they fit
    cells = [
        [Paragraph(v, ParagraphStyle("kv", fontSize=size, leading=size + 5,
                                     fontName="Helvetica-Bold",
                                     textColor=c, spaceAfter=2)),
         Paragraph(l, ParagraphStyle("kl", fontSize=8, leading=10,
                                     fontName="Helvetica", textColor=MUTED))]
        for l, v, c in label_vals
    ]
    col_w = 18.5 / n
    t = Table([cells], colWidths=_w([f"{col_w:.1f}cm"] * n))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), CARD),
        ("VALIGN",     (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",  (0,0),(-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1), 4),
        ("GRID", (0,0),(-1,-1), 0.5, BORDER),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return t


def _data_table(header: list, rows: list, styles, col_widths=None):
    data = [[Paragraph(str(h), ParagraphStyle("th", fontSize=8, fontName="Helvetica-Bold",
                                               textColor=CYAN))
             for h in header]]
    for row in rows:
        data.append([Paragraph(str(c), ParagraphStyle("td", fontSize=8, fontName="Helvetica",
                                                        textColor=MUTED))
                     for c in row])
    n = len(header)
    cw = col_widths or [f"{18.5/n:.1f}cm"] * n
    t  = Table(data, colWidths=_w(cw), repeatRows=1)
    ts = [
        ("BACKGROUND",   (0,0),(-1,0),  HexColor("#162032")),
        ("BACKGROUND",   (0,1),(-1,-1), CARD),
        ("GRID",         (0,0),(-1,-1), 0.5, BORDER),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[CARD, HexColor("#0D1525")]),
    ]
    t.setStyle(TableStyle(ts))
    return t


def _section_rule(styles):
    return HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=6, spaceBefore=4)


def _bullet(text, styles):
    return Paragraph(f"• {text}", styles["bullet"])


# ── Section builders ──────────────────────────────────────────────────────────

def _section_overview(df, styles) -> list:
    nums = _num_cols(df); cats = _cat_cols(df)
    total_rows, total_cols = df.shape
    missing = df.isnull().sum().sum()
    miss_pct = round(missing / (total_rows*total_cols)*100, 1)
    dup = int(df.duplicated().sum())

    score = 100
    score -= min(40, miss_pct*2)
    score -= min(20, (dup/total_rows*100)*2 if total_rows else 0)
    score  = max(0, round(score))
    score_color = GREEN if score>=80 else AMBER if score>=50 else RED

    els = [
        Paragraph("1. Dataset Overview", styles["h1"]),
        _section_rule(styles),
        _kpi_row([
            ("Total Records",  f"{total_rows:,}",   CYAN),
            ("Columns",        f"{total_cols}",      CYAN),
            ("Numeric",        f"{len(nums)}",       HexColor("#7C3AED")),
            ("Categorical",    f"{len(cats)}",       HexColor("#7C3AED")),
            ("Missing %",      f"{miss_pct}%",       AMBER if miss_pct>5 else GREEN),
            ("Duplicates",     f"{dup}",             RED if dup else GREEN),
            ("Quality Score",  f"{score}/100",       score_color),
        ], styles),
        Spacer(1, 10),
    ]
    return els


def _section_quality(df, styles) -> list:
    total_rows = len(df)
    rows = []
    for col in df.columns:
        miss = int(df[col].isnull().sum())
        miss_p = round(miss/total_rows*100,1) if total_rows else 0
        n_uniq = int(df[col].nunique())
        status = "Clean" if miss==0 else ("High" if miss_p>20 else "Low")
        rows.append([col, str(df[col].dtype), f"{miss} ({miss_p}%)", str(n_uniq), status])

    els = [
        Paragraph("2. Data Quality Report", styles["h1"]),
        _section_rule(styles),
        _data_table(
            ["Column","Type","Missing","Unique Values","Status"],
            rows,
            styles,
            col_widths=["5cm","2.5cm","3cm","3cm","3cm"],
        ),
        Spacer(1, 8),
    ]
    return els


def _section_stats(df, styles) -> list:
    nums = _num_cols(df)
    if not nums:
        return [Paragraph("3. Statistical Summary", styles["h1"]),
                Paragraph("No numeric columns found.", styles["body"])]

    rows = []
    for col in nums:
        s = df[col].dropna()
        q1,q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3-q1
        outliers = int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum())
        rows.append([
            col[:22],
            _fmt(s.mean()), _fmt(s.median()), _fmt(s.std()),
            _fmt(s.min()), _fmt(s.max()),
            f"{s.skew():.2f}", str(outliers),
        ])

    els = [
        Paragraph("3. Statistical Summary", styles["h1"]),
        _section_rule(styles),
        _data_table(
            ["Column","Mean","Median","Std Dev","Min","Max","Skew","Outliers"],
            rows, styles,
            col_widths=["3.5cm","2cm","2cm","2cm","2cm","2cm","1.5cm","2cm"],
        ),
        Spacer(1, 8),
    ]
    return els


def _section_correlations(df, styles) -> list:
    nums = _num_cols(df)
    if len(nums) < 2:
        return []

    corr = df[nums].corr().round(3)
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    pairs = upper.stack().reset_index()
    pairs.columns=["A","B","r"]
    pairs = pairs.dropna(subset=["r"])
    pairs["abs"] = pairs["r"].abs()
    top = pairs.sort_values("abs",ascending=False).head(10)

    rows = []
    for _, row in top.iterrows():
        a = abs(row["r"])
        strength = "Strong" if a > 0.7 else "Moderate" if a > 0.4 else "Weak"
        direction = "Positive" if row["r"] > 0 else "Negative"
        rows.append([str(row["A"])[:18], str(row["B"])[:18], str(row["r"]), strength, direction])

    els = [
        Paragraph("4. Correlation Analysis", styles["h1"]),
        _section_rule(styles),
        _data_table(
            ["Column A","Column B","r","Strength","Direction"],
            rows, styles,
            col_widths=["4cm","4cm","2.5cm","3cm","4cm"],
        ),
        Spacer(1, 8),
    ]
    return els


def _section_decisions(decisions: dict, score: int, styles) -> list:
    if not decisions:
        return []

    score_color = GREEN if score>=70 else AMBER if score>=40 else RED
    readiness   = "High" if score>=70 else "Medium" if score>=40 else "Low"

    priority_color = {"HIGH": RED, "MEDIUM": AMBER, "INFO": GREEN, "LOW": MUTED}

    els = [
        Paragraph("5. Decision Support Summary", styles["h1"]),
        _section_rule(styles),
        _kpi_row([
            ("Readiness Score", f"{score}/100", score_color),
            ("Readiness Level", readiness,       score_color),
            ("Immediate Actions",str(len(decisions.get("actions",[]))), RED),
            ("Risks",           str(len(decisions.get("risks",[]))),    AMBER),
            ("Opportunities",   str(len(decisions.get("opportunities",[]))), GREEN),
        ], styles),
        Spacer(1, 10),
    ]

    for section, label in [("actions","Immediate Actions"),
                            ("risks","Risks"),
                            ("watch","Watch List"),
                            ("opportunities","Opportunities")]:
        items = decisions.get(section, [])
        if not items:
            continue
        els.append(CondPageBreak(3*cm))
        els.append(Paragraph(label, styles["h2"]))
        for item in items:
            pc = priority_color.get(item["priority"], MUTED)
            els.append(KeepTogether([
                Paragraph(
                    f"<font color='#{pc.hexval()[2:]}'>&bull;</font>  <b>{escape(str(item['title']))}</b>",
                    ParagraphStyle("dc_title", fontSize=9, fontName="Helvetica-Bold",
                                   textColor=WHITE, spaceAfter=3, leftIndent=6)),
                Paragraph(escape(str(item["body"])),
                    ParagraphStyle("dc_body", fontSize=8, fontName="Helvetica",
                                   textColor=MUTED, spaceAfter=8, leftIndent=18, leading=13)),
            ]))
        els.append(Spacer(1, 4))

    return els


# ── Page background canvas ────────────────────────────────────────────────────

def _dark_canvas(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    # Header stripe
    canvas.setFillColor(SURFACE)
    canvas.rect(0, A4[1]-2*cm, A4[0], 2*cm, fill=1, stroke=0)
    # Cyan accent line
    canvas.setStrokeColor(CYAN)
    canvas.setLineWidth(1.5)
    canvas.line(0, A4[1]-2*cm, A4[0], A4[1]-2*cm)
    # Footer
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(2*cm, 0.6*cm, "DataVista AI  ·  Confidential Report")
    canvas.drawRightString(A4[0]-2*cm, 0.6*cm, f"Page {doc.page}")
    canvas.restoreState()


# ── Public generate function ──────────────────────────────────────────────────

def generate_pdf(df: pd.DataFrame, decisions: dict = None, score: int = 75) -> bytes:
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.4*cm, bottomMargin=1.8*cm,
    )
    styles = _make_styles()
    story  = []

    # Cover
    story.append(_header_table(df, styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Automated data analysis report for a dataset with "
        f"{len(df):,} rows and {len(df.columns)} columns.",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 14))

    # Sections
    story += _section_overview(df, styles)
    story.append(Spacer(1, 10))

    story.append(CondPageBreak(4*cm))
    story += _section_quality(df, styles)
    story.append(Spacer(1, 10))

    story.append(CondPageBreak(4*cm))
    story += _section_stats(df, styles)
    story.append(Spacer(1, 10))

    story.append(CondPageBreak(4*cm))
    story += _section_correlations(df, styles)
    story.append(Spacer(1, 10))

    if decisions:
        story.append(CondPageBreak(5*cm))
        story += _section_decisions(decisions, score, styles)

    doc.build(story, onFirstPage=_dark_canvas, onLaterPages=_dark_canvas)
    return buf.getvalue()
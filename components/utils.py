import numpy as np
import pandas as pd


def get_num_cols(df: pd.DataFrame) -> list:
    """Truly numeric columns. Binary columns (like 0/1 flags) are excluded."""
    nums = df.select_dtypes(include=np.number).columns
    return [c for c in nums if df[c].nunique(dropna=True) > 2]


def get_cat_cols(df: pd.DataFrame) -> list:
    """Text/category/bool columns, plus numeric columns with 2 or fewer unique values."""
    cats = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    nums = df.select_dtypes(include=np.number).columns
    binary = [c for c in nums if df[c].nunique(dropna=True) <= 2]
    return cats + binary

def assess_data(df: pd.DataFrame) -> dict:
    """Checks if a dataset is safe to chart. Returns a level and a list of issues.
    level: 'clean' | 'warning' | 'blocked'
    """
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        return {"level": "blocked", "score": 0, "issues": ["The dataset is empty."]}

    issues = []
    missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100
    dup_count = int(df.duplicated().sum())
    dup_pct = dup_count / n_rows * 100

    # Same formula as the Data Health Dashboard (keep them in sync)
    score = max(0, round(100 - min(40, missing_pct * 2) - min(20, dup_pct * 2)))

    # 1. Duplicate rows
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows ({dup_pct:.1f}%) will inflate counts and averages.")

    # 2. Columns that are mostly empty
    for c in df.columns:
        pct = df[c].isnull().mean() * 100
        if pct > 20:
            issues.append(f"'{c}' is {pct:.0f}% empty, so charts using it cover only part of the data.")

    # 3. Inconsistent spelling / case in text columns
    for c in df.select_dtypes(include=["object", "string"]).columns:
        raw = df[c].dropna().astype(str)
        if raw.nunique() > raw.str.strip().str.lower().nunique():
            issues.append(f"'{c}' has inconsistent spelling or case (e.g. 'Sales' vs 'sales '), "
                          f"so one category will show up as several.")

    # 4. Extreme outliers that stretch chart axes
    for c in get_num_cols(df):
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        extreme = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
        if extreme > 0:
            issues.append(f"'{c}' has {extreme} extreme outliers that will squash the rest of the chart.")

    # Decide the level
    if score < 50 or dup_pct > 30 or missing_pct > 40:
        level = "blocked"
    elif issues:
        level = "warning"
    else:
        level = "clean"

    return {"level": level, "score": score, "issues": issues}
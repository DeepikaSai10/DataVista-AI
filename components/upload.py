import streamlit as st
import pandas as pd


def upload_section():
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file to begin analysis",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        return None

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df.columns = [str(c) for c in df.columns]   # e.g. year headers 2021 -> "2021"
        return df
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None
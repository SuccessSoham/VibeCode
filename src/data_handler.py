# src/data_handler.py
import pandas as pd
import io
import streamlit as st

@st.cache_data # Cache data loading to prevent re-reading on every rerun
def load_csv_data(uploaded_file) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.
    """
    try:
        # Read the file into a BytesIO object first
        file_content = io.BytesIO(uploaded_file.getvalue())
        df = pd.read_csv(file_content)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def get_dataframe_info(df: pd.DataFrame) -> str: [12]
    """
    Generates a string summary of the DataFrame's schema and basic statistics.
    This provides crucial context for the LLM.
    """
    if df.empty:
        return "The DataFrame is empty."

    info_buffer = io.StringIO()
    df.info(buf=info_buffer, verbose=True, show_counts=True)
    df_info_str = info_buffer.getvalue()

    # Add descriptive statistics for numerical columns
    desc_buffer = io.StringIO()
    df.describe().to_csv(desc_buffer)
    desc_str = desc_buffer.getvalue()

    # Add value counts for top categorical columns (up to a limit)
    categorical_info = ""
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() < 20 and df[col].nunique() > 1: # Limit for reasonable display and exclude single-value columns
            categorical_info += f"\nValue counts for '{col}':\n{df[col].value_counts().to_string()}\n"

    # Also include head for schema understanding
    head_buffer = io.StringIO()
    df.head().to_csv(head_buffer)
    head_str = head_buffer.getvalue()

    return (
        f"DataFrame Info:\n```\n{df_info_str}\n```\n\n"
        f"First 5 rows (for schema reference):\n```\n{head_str}\n```\n\n"
        f"Descriptive Statistics:\n```\n{desc_str}\n```\n\n"
        f"Categorical Column Insights:\n{categorical_info}"
    )
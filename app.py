import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.code_executor import execute_code
from dotenv import load_dotenv
import time

load_dotenv() # Load environment variables from .env file
HF_TOKEN = os.getenv("HF_TOKEN")

# Import new utility modules from src

# Set page config
st.set_page_config(page_title="📊Data Analytics Agent", layout="wide")

# Custom CSS for dark mode and styling
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode", value=True)
if dark_mode:
    st.markdown("""
        <style>
            body {
                background: linear-gradient(to bottom, #1c1c1c, #2e2e2e);
                color: #f0f0f0;
                font-family: 'Segoe UI', sans-serif;
            }
            .stApp {
                background: linear-gradient(to bottom, #1c1c1c, #2e2e2e);
                color: #f0f0f0;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 1.1rem;
                font-weight: bold;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 20px;
            }
            .stTabs [data-baseweb="tab-list"] button {
                background-color: #333;
                color: #f0f0f0;
                border-radius: 8px;
                padding: 10px 20px;
                transition: all 0.2s ease-in-out;
            }
            .stTabs [data-baseweb="tab-list"] button:hover {
                background-color: #555;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                background-color: #007bff;
                color: white;
                border-bottom: 2px solid #007bff;
            }
            .stSpinner > div > div {
                border-top-color: #007bff;
            }
            .stButton > button {
                background-color: #007bff;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 1rem;
                font-weight: bold;
                transition: all 0.2s ease-in-out;
            }
            .stButton > button:hover {
                background-color: #0056b3;
            }
            .stAlert {
                border-radius: 8px;
            }
            .stTextInput > div > div > input {
                border-radius: 8px;
                border: 1px solid #555;
                background-color: #2e2e2e;
                color: #f0f0f0;
            }
            .stTextArea > div > div > textarea {
                border-radius: 8px;
                border: 1px solid #555;
                background-color: #2e2e2e;
                color: #f0f0f0;
            }
            .stSelectbox > div > div {
                border-radius: 8px;
                border: 1px solid #555;
                background-color: #2e2e2e;
                color: #f0f0f0;
            }
            .stSlider > div > div > div {
                background-color: #007bff;
            }
            .stSlider > div > div > div > div[data-testid="stThumbValue"] {
                color: #007bff;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #007bff;
            }
        </style>
    """, unsafe_allow_html=True)

# --- Global Data Loading and Filtering (for Tabs 1-3) ---
# Sidebar: Upload CSV for main dashboard
st.sidebar.header("📁 Upload Data for Dashboard")
dashboard_uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key="dashboard_uploader")

# Load default or uploaded data for dashboard
default_path = "data/business_metrics.csv"

if "global_df" not in st.session_state:
    st.session_state.global_df = None

if dashboard_uploaded_file:
    st.session_state.global_df = pd.read_csv(dashboard_uploaded_file)
elif os.path.exists(default_path) and st.session_state.global_df is None:
    st.session_state.global_df = pd.read_csv(default_path)

if st.session_state.global_df is not None:
    # Convert Month column to datetime
    if "Month" in st.session_state.global_df.columns:
        st.session_state.global_df["Month"] = pd.to_datetime(st.session_state.global_df["Month"], format="%b-%Y", errors='coerce')

    # Sidebar filters
    st.sidebar.header("🔍 Filter Criteria (for Dashboard Tabs)")

    # Timeline filter
    start_date = end_date = None
    if "Month" in st.session_state.global_df.columns and pd.api.types.is_datetime64_any_dtype(st.session_state.global_df["Month"]):
        min_date = st.session_state.global_df["Month"].min().date()
        max_date = st.session_state.global_df["Month"].max().date()
        start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

    # Numeric filters
    numeric_cols = st.session_state.global_df.select_dtypes(include='number').columns
    revenue_range = expenses_range = satisfaction_range = (0, 0)

    if "Revenue" in numeric_cols:
        revenue_range = st.sidebar.slider("Revenue range:",
            int(st.session_state.global_df["Revenue"].min()), int(st.session_state.global_df["Revenue"].max()),
            (int(st.session_state.global_df["Revenue"].min()), int(st.session_state.global_df["Revenue"].max())),
            key="revenue_slider"
        )

    if "Expenses" in numeric_cols:
        expenses_range = st.sidebar.slider("Expenses range:",
            int(st.session_state.global_df["Expenses"].min()), int(st.session_state.global_df["Expenses"].max()),
            (int(st.session_state.global_df["Expenses"].min()), int(st.session_state.global_df["Expenses"].max())),
            key="expenses_slider"
        )

    if "Customer_Satisfaction" in numeric_cols:
        satisfaction_range = st.sidebar.slider("Customer Satisfaction range:",
            int(st.session_state.global_df["Customer_Satisfaction"].min()), int(st.session_state.global_df["Customer_Satisfaction"].max()),
            (int(st.session_state.global_df["Customer_Satisfaction"].min()), int(st.session_state.global_df["Customer_Satisfaction"].max())),
            key="satisfaction_slider"
        )

    # Apply filters
    filtered_df = st.session_state.global_df.copy()
    if start_date and end_date:
        filtered_df = filtered_df[(st.session_state.global_df["Month"] >= start_date) & (st.session_state.global_df["Month"] <= end_date)]

    if "Revenue" in numeric_cols:
        filtered_df = filtered_df[filtered_df["Revenue"].between(*revenue_range)]

    if "Expenses" in numeric_cols:
        filtered_df = filtered_df[filtered_df["Expenses"].between(*expenses_range)]

    if "Customer_Satisfaction" in numeric_cols:
        filtered_df = filtered_df[filtered_df["Customer_Satisfaction"].between(*satisfaction_range)]


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "📈 Data Analytics", "📉 Data Visualization", "💬 Chat with Data"])

# --- Tab 1: Data Preview ---
with tab1:
    st.header("📊 Data Preview")
    if st.session_state.global_df is not None:

        st.subheader("DataFrame")
        st.dataframe(st.session_state.global_df)
        st.write("Number of Rows:", len(st.session_state.global_df))
        st.write("Number of Columns:", len(st.session_state.global_df.columns))

        st.subheader("Metadata")
        st.write("Column Names:", list(st.session_state.global_df.columns))
        st.write("Data Types:")
        st.write(st.session_state.global_df.dtypes)
        st.write("Missing Values:")
        st.write(st.session_state.global_df.isnull().sum())
        st.write("Duplicate Rows:", st.session_state.global_df.duplicated().sum())

        st.subheader("Statistical Summary")
        st.write("Mean:")
        st.write(st.session_state.global_df.select_dtypes(include='number').mean())
        st.write("Median:")
        st.write(st.session_state.global_df.select_dtypes(include='number').median())
        st.write("Mode:")
        st.write(st.session_state.global_df.select_dtypes(include='number').mode().iloc[0])
        st.write("Standard Deviation:")
        st.write(st.session_state.global_df.select_dtypes(include='number').std())
        st.write("Variance:")
        st.write(st.session_state.global_df.select_dtypes(include='number').var())
        st.write("Quartiles:")
        st.write(st.session_state.global_df.select_dtypes(include='number').quantile([0.25, 0.5, 0.75]))
    else:
        st.info("Please upload a CSV file in the sidebar to see data preview.")

# --- Tab 2: Data Analytics ---
with tab2:
    st.header("📈 Data Analytics")
    if st.session_state.global_df is not None:
        st.subheader("Data Quality Metrics")
        completeness = 1 - st.session_state.global_df.isnull().mean()
        st.write("Completeness:")
        st.write(completeness)

        st.subheader("Outlier Detection")
        for col in numeric_cols:
            q1 = st.session_state.global_df[col].quantile(0.25)
            q3 = st.session_state.global_df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = st.session_state.global_df[(st.session_state.global_df[col] < q1 - 1.5 * iqr) | (st.session_state.global_df[col] > q3 + 1.5 * iqr)]
            st.write(f"{col}: {len(outliers)} outliers")

        st.subheader("Recommendations")
        st.write("- Consider imputing missing values.")
        st.write("- Investigate outliers for potential data entry errors.")
        st.write("- Ensure consistent data formats across columns.")
    else:
        st.info("Please upload a CSV file in the sidebar to see data analytics.")

# --- Tab 3: Data Visualization ---
with tab3:
    st.header("📉 Data Visualization")
    if st.session_state.global_df is not None:
        st.subheader("Recommended Charts")

        if "Month" in st.session_state.global_df.columns and "Revenue" in st.session_state.global_df.columns and "Expenses" in st.session_state.global_df.columns:
            fig, ax = plt.subplots()
            filtered_df.plot(x="Month", y=["Revenue", "Expenses"], ax=ax, marker='o')
            st.pyplot(fig)

        if "Month" in st.session_state.global_df.columns and "Customer_Satisfaction" in st.session_state.global_df.columns:
            fig2, ax2 = plt.subplots()
            sns.lineplot(data=filtered_df, x="Month", y="Customer_Satisfaction", marker='o', ax=ax2)
            st.pyplot(fig2)

        st.subheader("Custom Visualization")
        chart_type = st.selectbox("Select chart type", ["Line", "Bar", "Scatter"], key="custom_chart_type")
        x_axis = st.selectbox("X-axis", st.session_state.global_df.columns, key="custom_x_axis")
        y_axis = st.selectbox("Y-axis", st.session_state.global_df.columns, key="custom_y_axis")

        if chart_type == "Line":
            fig3, ax3 = plt.subplots()
            sns.lineplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax3)
            st.pyplot(fig3)
        elif chart_type == "Bar":
            fig4, ax4 = plt.subplots()
            sns.barplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax4)
            st.pyplot(fig4)
        elif chart_type == "Scatter":
            fig5, ax5 = plt.subplots()
            sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax5)
            st.pyplot(fig5)
    else:
        st.info("Please upload a CSV file in the sidebar to see data visualizations.")

with tab4:
    st.header("💬 Chat with Data")
    st.write("Ask questions about your data using natural language. The AI will answer using open source models.")

    # Load Qwen or other Hugging Face model (cached)
    @st.cache_resource
    def load_hf_model():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", token=HF_TOKEN)
        return tokenizer, model

    tokenizer, model = load_hf_model()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What do you want to know about your data?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.global_df is None:
                st.warning("Please upload a CSV file in the sidebar to chat with data.")
                st.session_state.messages.append({"role": "assistant", "content": "Please upload a CSV file in the sidebar to chat with data."})
            else:
                # Prepare context for the model
                df_info = str(st.session_state.global_df.head(10))
                input_text = f"You are a data analyst. Here is a sample of the dataset:\n{df_info}\n\nUser question: {prompt}\n\nAnswer in natural language."
                with st.spinner("Generating answer..."):
                    input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
                    outputs = model.generate(**input_ids, max_new_tokens=1024)
                    raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove prompt context from output if present
                # Find the last occurrence of 'Answer in natural language.' and show only what comes after
                split_token = "Answer in natural language."
                if split_token in raw_answer:
                    answer = raw_answer.split(split_token, 1)[-1].strip()
                else:
                    answer = raw_answer.strip()
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})



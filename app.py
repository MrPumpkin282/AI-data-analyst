from datetime import datetime
import pandas as pd
import streamlit as st
from charting import generate_chart
from data_utils import get_schema_description, preprocess_dates, run_analysis_plan
from llm_agents import call_explainer_llm, call_planner_llm
from styles import CUSTOM_CSS

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>AI Data Analyst Agent</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Upload any CSV file to get started")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = preprocess_dates(df)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    with st.expander("Preview Data", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        st.dataframe(df.head(100), width='stretch', height=300)
        st.markdown("**Numeric Columns Summary:**")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), width='stretch')

    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            " Ask a question about your data",
            placeholder="e.g., Which product category had the highest sales last year?",
            label_visibility="collapsed"
        )
    with col2:
        analyze_button = st.button("Analyze", type="primary", width='stretch')

    if analyze_button and question.strip():
        with st.spinner("Analyzing data..."):
            schema_text = get_schema_description(df)
            try:
                plan, planner_reasoning = call_planner_llm(schema_text, question)
            except Exception as e:
                st.error(f" Planner agent failed: {e}")
                st.stop()

            with st.expander(" Analysis Plan", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.json(plan)
                with col2:
                    if planner_reasoning:
                        st.markdown("**AI Reasoning Process:**")
                        st.text_area("Planner Reasoning", planner_reasoning[:1000] + "..." if len(planner_reasoning) > 1000 else planner_reasoning,
                                     height=200, label_visibility="collapsed")

            try:
                result_df = run_analysis_plan(df, plan)
            except Exception as e:
                st.error(f" Error running analysis: {e}")
                st.info(" Tip: Try rephrasing your question or be more specific about the columns you want to analyze.")
                st.stop()

        st.markdown("---")
        st.markdown("## Results")

        chart_buf = generate_chart(result_df, plan)
        if chart_buf is not None:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(chart_buf)
            with col2:
                st.markdown("### Key Metrics")
                target_col = plan.get("target_column")
                if target_col in result_df.columns:
                    st.metric("Total", f"{result_df[target_col].sum():,.2f}")
                    st.metric("Average", f"{result_df[target_col].mean():,.2f}")
                    st.metric("Max", f"{result_df[target_col].max():,.2f}")
                    st.metric("Min", f"{result_df[target_col].min():,.2f}")
        else:
            st.info("Chart generation skipped (not applicable for this query)")

        with st.expander("Results Table", expanded=True):
            st.dataframe(result_df, width='stretch', height=400)
            csv_data = result_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with st.spinner("Generating insights..."):
            result_df_serializable = result_df.copy()
            for col in result_df_serializable.columns:
                if pd.api.types.is_datetime64_any_dtype(result_df_serializable[col]):
                    result_df_serializable[col] = result_df_serializable[col].astype(str)
                elif pd.api.types.is_period_dtype(result_df_serializable[col]):
                    result_df_serializable[col] = result_df_serializable[col].astype(str)
            result_summary = result_df_serializable.to_dict(orient="records")

            explanation, explainer_reasoning = call_explainer_llm(
                question=question,
                plan=plan,
                result_summary=result_summary,
            )

        st.markdown("---")
        st.markdown("## AI Insights")
        st.markdown(f'<div class="insight-box">{explanation}</div>', unsafe_allow_html=True)
        if explainer_reasoning:
            with st.expander(" AI Reasoning Process", expanded=False):
                st.text_area("Explainer Reasoning", explainer_reasoning[:1500] + "..." if len(explainer_reasoning) > 1500 else explainer_reasoning,
                             height=300, label_visibility="collapsed")

import importlib_metadata 
import streamlit as st
import streamlit as st
from analyzer import analyze_dataset
from llm_service import get_ai_insights

st.title("AI Data Processor")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    df, summary, err = analyze_dataset(file)
    if err:
        st.error(err)
    else:
        st.write("### Data Preview", df.head())
        st.write("### Statistics", summary["stats"])
        
        if st.button("Generate AI Insights"):
            with st.spinner("Analyzing..."):
                st.write(get_ai_insights(summary))
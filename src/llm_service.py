import streamlit as st
import google.generativeai as genai

# 1. Get the API Key from Streamlit Secrets (secrets.toml)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = None

# 2. Configure the AI
if api_key:
    genai.configure(api_key=api_key)

def get_ai_insights(summary):
    if not api_key:
        return "Error: Google API Key missing. Check .streamlit/secrets.toml"

    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""
    You are a Senior Data Analyst. 
    Analyze the following dataset summary and provide 3 actionable business insights.
    
    Data Summary:
    {summary}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"
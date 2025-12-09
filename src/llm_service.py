import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the API
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_ai_insights(summary):
    if not api_key:
        return "Error: Google API Key missing."

    model = genai.GenerativeModel('models/gemini-flash-latest')
    
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
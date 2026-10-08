import json
import os
from typing import Dict, List, Optional, Union
import google.generativeai as genai
from pydantic import BaseModel, Field
try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]


class InsightsSchema(BaseModel):
    """Structured insights schema enforced via Gemini JSON mode."""
    summary: str = Field(description="Executive summary of the dataset characteristics and patterns.")
    risks: List[str] = Field(description="List of key data risks, anomalies, or quality concerns identified.")
    recommendations: List[str] = Field(description="List of actionable business and data recommendations.")


class LLMConfigurationError(Exception):
    """Raised when the LLM service configuration or credentials are missing/invalid."""
    pass


class LLMGenerationError(Exception):
    """Raised when the LLM service fails to generate structured insights."""
    pass


def get_api_key() -> Optional[str]:
    """Retrieve Google API Key from Streamlit secrets or environment variables."""
    if st is not None:
        try:
            if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"])
        except Exception:
            pass
    return os.environ.get("GOOGLE_API_KEY")


def get_ai_insights(summary: Union[Dict[str, object], str]) -> Dict[str, Union[str, List[str]]]:
    """
    Generate structured business insights from aggregated dataset summary
    using Gemini's JSON mode and schema enforcement.

    Args:
        summary: Aggregated statistical summary dictionary or string representation.

    Returns:
        Structured dictionary matching InsightsSchema:
        {
            "summary": str,
            "risks": list[str],
            "recommendations": list[str]
        }

    Raises:
        LLMConfigurationError: If the Google API key is missing.
        LLMGenerationError: If model invocation or response parsing fails.
    """
    api_key = get_api_key()
    if not api_key:
        raise LLMConfigurationError(
            "Google API Key missing. Please configure 'GOOGLE_API_KEY' in .streamlit/secrets.toml or as an environment variable."
        )

    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=InsightsSchema,
    )

    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        generation_config=generation_config,
    )

    formatted_summary = (
        json.dumps(summary, indent=2)
        if isinstance(summary, (dict, list))
        else str(summary)
    )

    prompt = f"""You are a Senior Data Analyst and Chief Strategy Advisor.
Analyze the following aggregated tabular dataset summary and provide structured insights:
1. An executive summary highlighting key distribution characteristics and patterns.
2. Key operational, statistical, or data quality risks.
3. Actionable strategic recommendations based on the findings.

Aggregated Data Summary:
{formatted_summary}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        data = json.loads(text)
        validated = InsightsSchema.model_validate(data)
        return {
            "summary": validated.summary,
            "risks": validated.risks,
            "recommendations": validated.recommendations,
        }
    except Exception as e:
        raise LLMGenerationError(f"Failed to generate structured insights: {e}") from e

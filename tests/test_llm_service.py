import json
import os
import unittest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from llm_service import (
    InsightsSchema,
    LLMConfigurationError,
    LLMGenerationError,
    get_ai_insights,
    get_api_key,
)
from pydantic import ValidationError


class TestLLMService(unittest.TestCase):
    def test_insights_schema_validation(self):
        valid_payload = {
            "summary": "Revenue increased by 15% with stable margins.",
            "risks": ["Supply chain delay in Q3", "High customer churn in segment B"],
            "recommendations": ["Diversify vendor base", "Launch targeted retention campaign"],
        }
        model = InsightsSchema.model_validate(valid_payload)
        self.assertEqual(model.summary, valid_payload["summary"])
        self.assertEqual(model.risks, valid_payload["risks"])
        self.assertEqual(model.recommendations, valid_payload["recommendations"])

        # Test invalid payload missing required fields
        with self.assertRaises(ValidationError):
            InsightsSchema.model_validate({"summary": "Incomplete data"})

    def test_get_api_key_resolution(self):
        mock_st = MagicMock()
        mock_st.secrets = {}
        with patch("llm_service.st", mock_st):
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-test-key"}, clear=False):
                self.assertEqual(get_api_key(), "env-test-key")

        mock_st_with_key = MagicMock()
        mock_st_with_key.secrets = {"GOOGLE_API_KEY": "secrets-test-key"}
        with patch("llm_service.st", mock_st_with_key):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(get_api_key(), "secrets-test-key")

        with patch("llm_service.st", None):
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(get_api_key())

    def test_missing_api_key_raises_configuration_error(self):
        with patch("llm_service.get_api_key", return_value=None):
            with self.assertRaises(LLMConfigurationError) as ctx:
                get_ai_insights({"rows": 100, "columns": ["a", "b"]})
            self.assertIn("Google API Key missing", str(ctx.exception))

    @patch("llm_service.get_api_key", return_value="fake-api-key")
    @patch("llm_service.genai.configure")
    @patch("llm_service.genai.GenerativeModel")
    def test_structured_ai_insights_success(self, mock_model_cls, mock_configure, mock_get_api_key):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "summary": "Overall normal distributions with slight right skew on profit.",
            "risks": ["Profit variance indicates potential volatility.", "Missing values in region column."],
            "recommendations": ["Impute missing regional data.", "Hedge risk for volatile products."],
        })
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_instance

        summary_payload = {
            "columns": ["sales", "profit"],
            "rows": 500,
            "stats": {"sales": {"mean": 120.0}},
        }
        result = get_ai_insights(summary_payload)

        mock_configure.assert_called_once_with(api_key="fake-api-key")
        self.assertIn("generation_config", mock_model_cls.call_args[1])
        gen_config = mock_model_cls.call_args[1]["generation_config"]
        self.assertEqual(gen_config.response_mime_type, "application/json")
        self.assertEqual(gen_config.response_schema, InsightsSchema)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"], "Overall normal distributions with slight right skew on profit.")
        self.assertEqual(len(result["risks"]), 2)
        self.assertEqual(len(result["recommendations"]), 2)

    @patch("llm_service.get_api_key", return_value="fake-api-key")
    @patch("llm_service.genai.configure")
    @patch("llm_service.genai.GenerativeModel")
    def test_structured_ai_insights_markdown_fence_stripping(self, mock_model_cls, mock_configure, mock_get_api_key):
        mock_response = MagicMock()
        mock_response.text = """```json
{
  "summary": "Clean dataset summary.",
  "risks": ["Low sample size"],
  "recommendations": ["Collect more samples"]
}
```"""
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_instance

        result = get_ai_insights("Summary string")
        self.assertEqual(result["summary"], "Clean dataset summary.")
        self.assertEqual(result["risks"], ["Low sample size"])
        self.assertEqual(result["recommendations"], ["Collect more samples"])

    @patch("llm_service.get_api_key", return_value="fake-api-key")
    @patch("llm_service.genai.configure")
    @patch("llm_service.genai.GenerativeModel")
    def test_structured_ai_insights_invalid_json_raises_generation_error(self, mock_model_cls, mock_configure, mock_get_api_key):
        mock_response = MagicMock()
        mock_response.text = "Not a valid JSON response"
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_instance

        with self.assertRaises(LLMGenerationError):
            get_ai_insights({"rows": 10})

    @patch("llm_service.get_api_key", return_value="fake-api-key")
    @patch("llm_service.genai.configure")
    @patch("llm_service.genai.GenerativeModel")
    def test_structured_ai_insights_api_exception_raises_generation_error(self, mock_model_cls, mock_configure, mock_get_api_key):
        mock_instance = MagicMock()
        mock_instance.generate_content.side_effect = RuntimeError("Quota exceeded or network error")
        mock_model_cls.return_value = mock_instance

        with self.assertRaises(LLMGenerationError) as ctx:
            get_ai_insights({"rows": 10})
        self.assertIn("Failed to generate structured insights", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

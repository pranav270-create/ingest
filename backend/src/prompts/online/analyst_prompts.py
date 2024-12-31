import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parents[2]))

from src.prompts.parser import text_cost_parser, structured_text_cost_parser 
from src.prompts.registry import PromptRegistry


@PromptRegistry.register("data_analyst")
class DataAnalyst:
    system = (
        "You are a data analysis assistant that helps select the most relevant analysis functions "
        "to answer user queries"
    )

    user = (
        "Based on the user's question: '{query}'\n\n"
        "And the following available analysis functions:\n"
        "{context}\n\n"
        "Select only the most relevant function names that are necessary to answer the query.\n\n"
        "Requirements:\n"
        "1. Only select functions that address the user's question\n"
        "2. Limit selections to essential functions (prefer fewer over more)\n"
        "3. If no functions are very clearly relevant, return an empty list\n\n"
        "Return your response as a list of function names only, without any explanation."
    )

    class DataModel(BaseModel):
        function_names: list[str] = Field(description="The list of functions to select to answer the user query.")

    @classmethod
    def format_prompt(cls, query: str, context: list[Any], top_k: int = 5) -> dict:
        context = context[:top_k]
        # Format the context list into a readable string
        formatted_context = "\n".join(
            f"Function: {item['function_name']}\n"
            f"Description: {item['description']}\n"
            f"Relevance Score: {item['score']:.2f}\n"
            for item in context
        )

        return {
            "system": cls.system,
            "user": cls.user.format(
                query=query,
                context=formatted_context
            )
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return structured_text_cost_parser(response, model)


@PromptRegistry.register("summarize_trends")
class SummarizeTrends:
    system = (
        "You are a data analysis expert who excels at synthesizing complex analytical results into "
        "clear, actionable insights. Your role is to examine multiple analysis outputs and create "
        "a coherent narrative that highlights key findings, relationships, and practical implications. "
        "Focus on statistical significance, effect sizes, and practical relevance while maintaining "
        "scientific rigor in your interpretations."
    )

    user = (
        "Analyze the following data analysis results:\n"
        "{data_result}\n\n"
        "Provide a concise summary that includes:\n"
        "1. Key Findings: Highlight the most significant relationships and effects discovered\n"
        "2. Statistical Context: Note relevant sample sizes, effect sizes, and confidence levels\n"
        "3. Practical Implications: Explain what these findings mean in practical terms\n"
        "4. Limitations: Mention any important caveats or limitations in the analysis\n"
        "5. Recommendations: Suggest potential actions or areas for further investigation\n\n"
        "Format your response in clear sections. Focus on insights that are:\n"
        "- Statistically meaningful (considering p-values and effect sizes)\n"
        "- Practically significant (having real-world impact)\n"
        "- Well-supported by the data (adequate sample sizes)\n\n"
        "If certain relationships are weak or inconclusive, acknowledge this rather than "
        "forcing interpretations. Maintain scientific objectivity while making the insights "
        "accessible to non-technical stakeholders."
    )

    @classmethod
    def format_prompt(cls, data_result: list[dict]) -> dict:
        # Format the analysis results into a readable string
        formatted_results = []
        for result in data_result:
            formatted_results.append(
                f"Analysis Function: {result['function_name']}\n"
                f"Description: {result['docstring']}\n"
                f"Results: {cls._format_analysis_data(result['analysis_data'])}\n"
            )

        return {
            "system": cls.system,
            "user": cls.user.format(
                data_result="\n\n".join(formatted_results)
            )
        }

    @staticmethod
    def _format_analysis_data(analysis_data: dict) -> str:
        """Helper method to format analysis data in a structured way"""
        # Convert the analysis data into a readable format
        formatted_parts = []

        def _format_value(value):
            if isinstance(value, (int, float)):
                return f"{value:.3f}" if isinstance(value, float) else str(value)
            if isinstance(value, dict):
                return "{" + ", ".join(f"'{k}': {_format_value(v)}" for k, v in value.items()) + "}"
            if isinstance(value, list):
                return "[" + ", ".join(_format_value(v) for v in value) + "]"
            return str(value)

        def _format_dict(d: dict, prefix="") -> list[str]:
            parts = []
            for key, value in d.items():
                if isinstance(value, dict):
                    parts.append(f"{prefix}{key}:")
                    parts.extend(_format_dict(value, prefix + "  "))
                else:
                    parts.append(f"{prefix}{key}: {_format_value(value)}")
            return parts

        formatted_parts.extend(_format_dict(analysis_data))
        return "\n".join(formatted_parts)

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)

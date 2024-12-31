from typing import Any

from src.prompts.parser import text_cost_parser


class GenerateSummaryEmailPrompt:
    """
    Prompt for the ExpertAgent to generate a summary email.
    """

    system = "You are an expert at biology, medicine, and health. You are scientifically rigorous."

    user_template = prompt = """Please analyze the following sleep metrics and provide personalized recommendations:

{time_period}\n
{consistency}\n
{bedtimes}\n
{duration}\n
{efficiency}\n
{stages}\n
{heart_metrics}\n

Based on this data, please provide:
1. Top 3 priorities for improving sleep quality
2. Specific, actionable recommendations for each priority
3. Optimal sleep schedule suggestion
4. Exercise timing recommendations
5. Recovery optimization strategies

Consider both the metrics and their trends when making recommendations."""

    @classmethod
    def format_prompt(
        cls, time_period: str, consistency: str, bedtimes: str, duration: str, efficiency: str, stages: str, heart_metrics: str
    ):
        return {
            "system": cls.system,
            "user": cls.user_template.format(
                time_period=time_period,
                consistency=consistency,
                bedtimes=bedtimes,
                duration=duration,
                efficiency=efficiency,
                stages=stages,
                heart_metrics=heart_metrics,
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)

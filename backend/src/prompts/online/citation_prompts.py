from typing import Any

from pydantic import BaseModel

from src.prompts.parser import structured_text_cost_parser, text_cost_parser


class TextChunkWithAttribution(BaseModel):
    text: str
    attribution: list[int]


class SentencePlanPrompt:
    """
    Prompt for the ExpertAgent to answer questions by breaking them into sub-questions and retrieving context.
    """

    system = "You are an expert at biology, medicine, and health. You are scientifically rigorous."

    user = (
        "Sources:\n{sentences}"
        "Draft a response to the following question by writing one sentence at time."
        " Each sentence should be attributed to one, two, or three sources."
        " To make an attribution, just provide the integers in a list."
        "\nQuestion:\n{question}"
    )

    class DataModel(BaseModel):
        text_chunks: list[TextChunkWithAttribution]

    @classmethod
    def format_prompt(cls, question, context: list[str]):
        sentences = "\n".join(context)
        return {
            "system": cls.system,
            "user": cls.user.format(question=question, sentences=sentences),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[list[TextChunkWithAttribution], float]:
        response, cost = structured_text_cost_parser(response, model)
        return response.text_chunks, cost


class ReviseDraftPrompt:
    system = (
        "You are an expert in biology, medicine, and health, and you are scientifically rigorous."
        " Your task is to refine a response to ensure clarity and coherence while preserving all citations exactly as they appear."  # noqa
    )
    user = (
        "You will be given a question and a draft response consisting of properly-cited sentences."
        " The citations are denoted in square brackets, like [1], [2], [3]."
        " Your response must use these citations verbatim, preserving each one without modification."
        " Structure the response thoughtfully, but do not remove citations."
        " Focus on crafting a cohesive answer that integrates each cited sentence effectively."
        " You are allowed to write sentences that are not from the pre-given list."
        " \n\nQuestion:\n{question}\nDraft:\n{draft}"
    )

    @classmethod
    def format_prompt(cls, question, draft: list[str]):
        import random

        draft = random.sample(draft, len(draft))
        draft = "\n".join(draft)
        return {
            "system": cls.system,
            "user": cls.user.format(question=question, draft=draft),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)

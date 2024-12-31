from typing import Any

from pydantic import BaseModel

from src.prompts.parser import structured_text_cost_parser, text_cost_parser
from src.retrieval.retrieve_new import FormattedScoredPoints


class AnswerQuestionPrompt:
    """
    Prompt for the ExpertAgent to answer questions by breaking them into sub-questions and retrieving context.
    """

    system = "You are an expert at biology, medicine, and health. You are scientifically rigorous."

    user_template = (
        "\nEvidence:\n{evidence}"
        "Answer the following question using the context provided."
        " Do not make assumptions."
        "\nQuestion:\n{question}"
    )

    @classmethod
    def format_prompt(cls, question, context: list[FormattedScoredPoints]):
        evidence = ""
        for item in context:
            title = item.ingestion.document_title
            evidence += f"{title}\n{item.raw_text}\n\n"

        return {
            "system": cls.system,
            "user": cls.user_template.format(question=question, evidence=evidence),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class ReWriteQueryPrompt:
    """
    Prompt for the ExpertAgent to answer questions by breaking them into sub-questions and retrieving context.
    """

    system = "You are an expert at biology, medicine, and health. You are scientifically rigorous."

    user_template = (
        "\nQuestion:\n{question}"
        "Rewrite the question to be more specific and focused. Only split the question into multiple parts if absolutely"
        " necessary ot improve retrieval. Avoid excessive subquestions that could slow down the question answering process."
    )

    class DataModel(BaseModel):
        rewritten_question: list[str]

    @classmethod
    def format_prompt(cls, question):
        return {
            "system": cls.system,
            "user": cls.user_template.format(question=question),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = structured_text_cost_parser(response, model)
        return response.rewritten_question, cost

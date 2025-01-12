import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry import PromptRegistry


class QuestionAnswerPair(BaseModel):
    question: str = Field(..., description="A question that can be addressed by this text chunk")
    answer: str = Field(..., description="The answer to the question")


@PromptRegistry.register("synthetic_qa_pair")
class SyntheticQAPairPrompt:
    system_prompt = (
        "You are an assistant specialized in RAG tasks."
        " The task is the following: given a document chunk, you will have to"
        " generate questions that can be asked by a user to retrieve information from a large documentary corpus."
    )

    user_prompt = (
        "The question should be relevant to the chunk, and should not be too specific"
        " or too general. The question should be about the subject of the chunk, and"
        " the answer needs to be found in the chunk."
        " Remember that the question is asked by a user to get some information from a"
        " large documentary corpus."
        " Generate a question that could be asked by a user without knowing the existence and the content of the corpus."
        " Also generate the answer to the question, which should be found in the"
        " document chunk.  \n"
        " Generate TWO pairs of questions and answers per chunk. Follow the schema\n"
        " Chunk: {chunk_text}\n"
    )

    class DataModel(BaseModel):
        questions: list[QuestionAnswerPair] = Field(
            ..., description="A list of two question-answer pairs that are based on the chunk of text"
        )

    @classmethod
    async def format_prompt(cls, base_model: BaseModel, read=None, **kwargs) -> tuple[str, str]:
        return cls.system_prompt, cls.user_prompt.format(chunk_text=base_model['input'])

    @classmethod
    def parse_response(cls, base_models: list[BaseModel], parsed_items: dict[str, BaseModel]) -> list[BaseModel]:
        for i, basemodel in enumerate(base_models):
            model_response = parsed_items.get(i)
            basemodel['synthetic_questions'] = model_response["questions"]
        return base_models
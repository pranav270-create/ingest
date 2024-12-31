import re
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.prompts.base_prompt import BasePrompt
from src.prompts.parser import structured_text_cost_parser, text_cost_parser


class SummarizePodcastPrompt:
    """ """

    system = (
        "You are an expert in health and science communication. Your task is to provide a concise and accurate summary of the"
        " podcast. Focus on the key scientific claims, the context in which they are made, and the implications of these claims."
    )

    user = (
        "You will be given a transcript of a podcast on health and science."
        " Your job is to summarize the podcast, highlighting the main scientific claims and their relevance."
        " Podcast Transcript:\n{podcast}"
    )

    @classmethod
    def format_prompt(cls, podcast):
        return {"system": cls.system, "user": cls.user.format(podcast=podcast)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class ClaimAndContext(BaseModel):
    claim: str = Field(..., description="The claim made in the podcast")
    conversation_chunk: str = Field(
        ..., description="A long chunk of text from the podcast trannscript that gives context for the claim"
    )


class PodcastToClaimsPrompt(BasePrompt):
    """
    This prompt uses structured output to extract the key scientific claims from a podcast transcript.
    """

    system = (
        "You are an expert in biology, medicine, and health. Your task is to extract key scientific claims from the podcast"
        " transcript. Focus on identifying specific claims related to science, medicine, and personal health, and provide"
        " context for each claim."
    )

    user = (
        "You will be given a transcript of a podcast on health and science."
        " Your job is to extract the key scientific claims made in the podcast."
        " For each claim, provide a long chunk of text from the podcast transcript that provides context for the claim."
        " Please output the text in the provided structure."
        " Podcast Transcript:\n{podcast}"
    )

    class DataModel(BaseModel):
        """
        DataModel for extracting thorough, scientifically rigorous information from a podcast transcript
        """

        features: list[ClaimAndContext] = Field(..., description="A list of claims and supporting context from a podcast")

    @classmethod
    def format_prompt(cls, podcast):
        return {"system": cls.system, "user": cls.user.format(podcast=podcast)}

    @staticmethod
    def parse_response(response: DataModel, model: str) -> tuple[list[str], list[str], float]:
        text, cost = structured_text_cost_parser(response, model)
        return [f.claim for f in text.features], [f.conversation_chunk for f in text.features], cost


class ExtractPodcastData(BasePrompt):
    """
    This prompt uses structured output to extract the key scientific claims from a podcast transcript.
    """

    system = (
        "You are an expert in biology, medicine, and health. Your task is to extract key scientific claims from the podcast"
        " transcript. Focus on identifying specific claims related to science, medicine, and personal health, and provide"
        " context for each claim."
    )

    user = (
        "You will be given a transcript of a podcast on health and science."
        " Your job is to extract the key scientific claims made in the podcast."
        " For each claim, provide a long chunk of text from the podcast transcript that provides context for the claim."
        " Please output the text in the provided structure."
        " Podcast Transcript:\n{podcast}"
    )

    class DataModel(BaseModel):
        """
        DataModel for extracting thorough, scientifically rigorous information from a podcast transcript
        """

        podcast_host_name: str = Field(..., description="The name of the host of the podcast")
        podcast_guest_name: str = Field(..., description="The name of the guest of the podcast")
        podcast_main_topic: str = Field(..., description="The main topic of the podcast")
        podcast_claims: list[str] = Field(..., description="A list of the key claims made in the podcast")

    @classmethod
    def format_prompt(cls, podcast):
        return {"system": cls.system, "user": cls.user.format(podcast=podcast)}

    @staticmethod
    def parse_response(response: DataModel, model: str) -> tuple[list[str], list[str], float]:
        text, cost = structured_text_cost_parser(response, model)
        return text.podcast_host_name, text.podcast_guest_name, text.podcast_main_topic, text.podcast_claims, cost


class SummarizeSearchResultsPrompt:
    system = (
        "You are an expert at analyzing scientific claims and related context. "
        "You will be given a scientific claim and multiple pieces of relevant context. "
        "Your job is to summarize the relevant context and explain how it relates to the claim."
    )

    entry_template = "Relevance Score: {score}\n" "Title: {title}\n" "Evidence: {evidence}\n"

    user_template = (
        "You will be given a claim and multiple pieces of evidence. The evidence has relevance scores between 0 and 1 where 0"
        " is irrelevant to the claim and 1 is completely relevant to the claim.\n"
        "Your job is to analyze the following evidence in relation to the claim. Explain if the evidence is in support of the"
        " claim or not.\n"
        "Claim: {claim}\n"
        "{entries}"
    )

    @classmethod
    def format_prompt(cls, claim: str, entries: list[dict[str, Any]]):
        formatted_entries = "\n".join(
            [
                cls.entry_template.format(evidence=e["input"], score=e["score"], title=e["ingestion"].document_title)
                for e in entries
            ]
        )
        return {"system": cls.system, "user": cls.user_template.format(claim=claim, entries=formatted_entries)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class OutlineBlogPrompt:
    system = (
        "You are an expert at biology, medicine, and health. You are scientifically rigorous."
        " You are an expert at writing science articles. You are a science writer."
        " You are creative and engaging, but also professional."
    )

    claim_template = "{claim}\nEvidence Summary: {evidence_summary}\n"

    user_template = (
        "You will be given a summary of a podcast between {host} and {guest},"
        " and a series of summaries of evidence relating to the scientific claims made in the podcast."
        " Your job is to outline a blog post that synthesizes this information in a way that is compelling and in a way that"
        " teaches the reader something new or useful."
        " The structure of your blog should create a compelling narrative."
        " Use the evidence summaries when you analyze the claims."
        "\nPodcast Summary:\n{podcast_summary}"
        "\nContext:\n{context}"
    )

    @classmethod
    def format_prompt(cls, host, guest, podcast_summary, claims, evidence_summaries):
        formatted_claims = []
        for i, (claim, evidence_summary) in enumerate(zip(claims, evidence_summaries), 1):
            formatted_claim = cls.claim_template.format(claim=claim, evidence_summary=evidence_summary)
            formatted_claims.append(f"Claim {i}:\n{formatted_claim}")
        context = "\n\n".join(formatted_claims)

        return {
            "system": cls.system,
            "user": cls.user_template.format(host=host, guest=guest, podcast_summary=podcast_summary, context=context),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class CritiqueOutlinePrompt:
    system = (
        "You are an expert at biology, medicine, and health. You are scientifically rigorous."
        " You are an expert at writing science articles. You are a science writer."
        " You are creative and engaging, but also professional."
    )

    user = (
        "You will be given an outline of a blog post and a series of summaries of evidence relating to the scientific claims."
        " Your job is to critique the outline and suggest changes to the outline."
        " You should suggest changes to the outline to improve the narrative and the flow of the blog post."
        " You should also suggest changes to the outline to improve the flow of the blog post."
        " You should also suggest changes to the outline to improve the flow of the blog post."
    )

    @classmethod
    def format_prompt(cls, question, entries):
        formatted_entries = "\n".join(f"Entry {i+1}:\n{cls.entry_template.format(**entry)}" for i, entry in enumerate(entries))
        return {"system": cls.system, "user": cls.user_template.format(question=question, formatted_entries=formatted_entries)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


science_system_prompt = "You are an expert at biology, medicine, and health. You are scientifically rigorous."

ingest_podcast_user_prompt = (
    "You will be given a transcript of a <podcast> between {host} and {guest}."
    " Your job is to state key the key scientific claims that are made in the <podcast>."
    " We care about the specific claims about science, medicine, and health."
    " Please enclose each claim in a separate <claim> tag."
    " <podcast> {podcast} </podcast>"
)


summarize_returned_entries_preamble = (
    "You will be given a scientific claim, some relevant context on the topic,"
    " and a score between 0 and 1, where 0 is completely irrelevant and 1 is completely relevant"
    " Your job is to summarize the relevant context, if it is relevant, and explain how it relates to the claim."
    "{entries}"
)

summarize_returned_entries_user_prompt = (
    "--------------------------------\n" "Claim: {question}\n" "Context: {context}\n" "Relevance Score: {score}\n"
)

synthesize_user_prompt = (
    "Your task is to write a long, scientific article responding to a {type_of_content} from {publisher_of_content}."
    " You will be given a set of scientific claims from the original {type_of_content} and a series of summaries of evidence"
    " relating to the scientific claims. Your article must synthesize all of this information in a way that is compelling and"
    " in a way that teaches the reader something new or useful. You can critique or support the original claims, but be specific"
    " in your analysis, and you must ground that evidence in science. The article should be written in a way that is accessible"
    " to a layperson, but it should be scientifically rigorous. The article should be at least 1000 words.\n{summaries}"
)


class WriterPrompt:
    """
    Prompt for the WriterAgent to generate questions based on the claim and perspective.
    """

    system = (
        "You are an expert at biology, medicine, and health. You are a reporter and want to write high-quality articles."
        " You help generate insightful questions on topics you are interested in reporting on."
    )

    user = (
        "You are writing a scientific article in which you investigate a claim made by a prominent voice on health and wellness."
        " You are interviewing an expert on health and wellness to assess the veracity of the claim."
        " Given the claim and some additional context from the podcast,"
        " pose an interesting, insightful question to the expert on the topic to uncover a deeper understanding of the truth."
        "\nHistory: {history}"
    )

    @classmethod
    def format_prompt(cls, history):
        return {"system": cls.system, "user": cls.user.format(history=history)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class WriterPromptStormmail:
    """
    Prompt for the WriterAgent to generate questions based on the claim and perspective.
    """

    system = "You are a reporter with deep expertise in medicine, and health."

    user = (
        "You are writing a scientific article in which you investigate a claim made by a prominent voice on health and wellness."
        " You are interviewing an expert on health and wellness to assess the veracity of the claim."
        " Given the claim,"
        " pose an interesting, insightful question to the expert on the topic to uncover a deeper understanding of the truth."
        "\n{claim}\n{history}"
    )

    @classmethod
    def format_prompt(cls, claim, history):
        if len(history) < 100:
            history = ""
        return {"system": cls.system, "user": cls.user.format(claim=claim, history=history)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class ExpertPrompt:
    """
    Prompt for the ExpertAgent to answer questions by breaking them into sub-questions and retrieving context.
    """

    system = "You are an expert at biology, medicine, and health. You are scientifically rigorous."

    entry_template = "Context: {context}\n" "Relevance Score: {score}\n"

    user_template = (
        "\nEvidence:\n{evidence}"
        "You are being interviewed by a science writer. They have asked you a question about a particular claim that was made by"
        " the host of a popular health podcast."
        " questions to deepen their understanding. You only make claims when you have evidence to support it. Answer the question"
        " given the evidence, be thorough and descriptive in your analysis."
        "\nClaim:\n{claim}"
        "\nQuestion:\n{question}"
    )

    @classmethod
    def format_prompt(cls, claim, question, entries: list[dict[str, Any]]):
        formatted_entries = "\n".join([cls.entry_template.format(context=e["input"], score=e["score"]) for e in entries])
        return {
            "system": cls.system,
            "user": cls.user_template.format(claim=claim, question=question, evidence=formatted_entries),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class EmailPromptStormmail:
    """
    Prompt for the Agent to generate a personalized note for a prospective client based on a recent popular health podcast.
    """

    system = (
        "You are a health optimization expert working at Apeiron Life, also known as AL, specializing in personalized"
        " communication for high-end clients. You have deep knowledge of health podcasts, current wellness trends,"
        " and the needs of busy professionals seeking to improve their health and performance."
    )

    user = (
        "Write a short, personalized note to a client with the following profile:"
        "{client_profile}"
        "Reference a recent podcast episode using this structure:\n"
        "1. Podcast Reference (1 short sentence):"
        "\nHost: {host_name}\nGuest: {guest_name}\nTopic: {topic}\nKey Claim: {claim}\n"
        "2. Validate and Expand (1-2 short sentences):"
        " Confirm the claim's validity and explain its relevance to the client's multi-generational health goals.\n"
        "3. Bridge to Services (1-2 short sentences):"
        "Highlight how your company's approach addresses the family's diverse needs and enhances the podcast advice.\n"
        "Additional Context:\n"
        "{conversation_history}\n"
        "Tone: Knowledgeable, personalized, and subtly persuasive. Don't be overly salesy. Don't write a call-to-action."
        " Tailor the message to address client profile.\n"
        "Maximum length: 200 words\n"
        "Note: Avoid typical email greetings or signatures. Focus on creating a concise message that resonates with"
        " the client profile. Do not make health recommendations and do not directly try to sell."
        "EXAMPLES:\n"
        "Here are two examples of the kind of note we're looking for. Notice how they are short and to the point and do not try"
        " to sell the individual directly. Use these as inspiration for style and structure, but create a unique message"
        " tailored to the given client profile and podcast information:\n"
        "Example 1:\n"
        "---\n"
        "Not sure if you caught the Peter Attia Drive or Huberman podcasts last week but a few really interesting points for your"
        " partners related to our initial conversation. Huberman had Dr. Charan Raganath on to talk about cognition."
        " Dr. Ragantath said many of the most important factors for **maintaining memory as we age** are related to general"
        " health: Sleep, exercise, social engagement, avoiding smoking, most food from unprocessed or minimally processed foods,"
        " maintaining a sense of purpose. Our POV: We 100% agree, as you might expect - and hold our members accountable to"
        " changes in these arenas. As you mentioned, it's one thing to hear it on a podcast -- it's quite another to have a"
        " personal team making it accessible and keeping each exec on track."
        "---\n"
        "Example 2:\n"
        "---\n"
        "Attia had Dr. Marty Makary on to talk about the medical system generally BUT they touched on the large preventative"
        " impact of Hormone Replacement Therapy (HRT) for postmenopausal women. While we don't perform HRT, we do refer out"
        " in cases where relevant. This is the advantage of having a personal 'Chief Health Officer' working across disciplines &"
        " keeping up with the trends, for all genders."
        "---\n"
        "Now, using the provided client profile and podcast information, create a new, unique note following a similar style and"
        " structure:\n"
    )

    @classmethod
    def format_prompt(cls, host_name, guest_name, topic, client_profile, claim, history):
        return {
            "system": cls.system,
            "user": cls.user.format(
                host_name=host_name,
                guest_name=guest_name,
                topic=topic,
                client_profile=client_profile,
                conversation_history=history,
                claim=claim,
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class OutlineDraftPrompt:
    """
    Prompt for the OutlineDraftAgent to create a draft outline based on the topic.
    """

    system = "You are an expert science writer. You create comprehensive and well-structured outlines for articles."

    user = """Create a draft outline for a blog post that is to be written in response to a summary of a podcast.\n
            Here is the format of your writing:
            1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate
            subsubsection title, and so on.
            2. Do not include other information.
            \nPodcast Summary: {topic}
            """

    @classmethod
    def format_prompt(cls, topic):
        return {"system": cls.system, "user": cls.user.format(topic=topic)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class OutlineDraftRevisionPrompt:
    """
    Prompt for the OutlineDraftRevisionAgent to revise the draft outline based on conversation history.
    """

    system = "You are an expert science writer. You revise outlines to improve narrative flow and coherence."

    user = """Create a revised outline for a blog post that is to be written in response to a summary of a podcast.\n
            Here is the format of your writing:
            1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate
             subsubsection title, and so on.
            2. Do not include other information.
            \nPodcast Summary: {topic}
            \nDraft Outline: {draft_outline}
            \nConversation History: {conversation_history}
            """

    @classmethod
    def format_prompt(cls, topic, draft_outline, conversation_history):
        return {
            "system": cls.system,
            "user": cls.user.format(topic=topic, draft_outline=draft_outline, conversation_history=conversation_history),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[list[str], float]:
        text, cost = text_cost_parser(response, model)
        sections = re.split(r"\n(?=# )", text.strip())
        sections = [section.strip() for section in sections if section.strip()]
        return sections, cost


class OutlinePrompt:
    """
    Prompt for the OutlinePromptAgent to create an outline based on a podcast summary and a critical conversation about it.
    """

    system = "You are an expert science writer. You revise outlines to improve narrative flow and coherence."

    user = """You are writing a blog post in response to a podcast.\n
            You have interviewed an expert on health and wellness to better understand the veracity of claims made by the podcast.
            Using the conversation history, create an outline for the blogpost that critically analyzes the claims made.
            Your goal is to find truth. Do not be afraid to disagree with the podcast.
            Here is the format of your writing:
            1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate
             subsubsection title, and so on.
            2. Do not include other information.
            \nPodcast Summary: {summary}
            \nConversation History: {conversation_history}
            """

    @classmethod
    def format_prompt(cls, summary, conversation_history):
        return {
            "system": cls.system,
            "user": cls.user.format(summary=summary, conversation_history=conversation_history),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[list[str], float]:
        text, cost = text_cost_parser(response, model)
        sections = re.split(r"\n(?=# )", text.strip())
        sections = [section.strip() for section in sections if section.strip()]
        return sections, cost


class SectionWriterPrompt:
    """
    Prompt for the SectionWriterAgent to write a section based on the outline and evidence.
    """

    system = "You are an expert science writer. You write detailed and engaging sections for blog posts."

    user = (
        "You are writing a section of a blog post based on the following outline section:\n"
        "{outline_section}\n"
        "Use the following evidence to support your writing. Each piece of evidence has a citation number."
        " Use these numbers to cite your sources in the text, like this: [1], [2], etc."
        "Evidence:\n{evidence}\n"
        "Write a detailed and engaging section of the blog post, making sure to:\n"
        "1. Stick to the topic of the outline section.\n"
        "2. Use the provided evidence, citing it appropriately with the given citation numbers.\n"
        "3. Maintain a critical and analytical tone. If there is a discrepancy between the claim and the evidence, note it.\n"
        "4. Ensure smooth transitions between ideas.\n"
        "5. Conclude the section in a way that leads into the next part of the outline.\n"
        "Your response should be in markdown format."
    )

    @classmethod
    def format_prompt(cls, outline_section, evidence):
        return {"system": cls.system, "user": cls.user.format(outline_section=outline_section, evidence=evidence)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class EditorPrompt:
    """
    Prompt for the EditorAgent to polish the concatenated draft sections into a polished article.
    """

    system = "You are a professional science editor. You polish and refine articles to ensure clarity, coherence, and engagement."

    user = (
        "You are a faithful text editor that is good at finding repeated information in the article and deleting them"
        " to make sure there is no repetition in the article. You won't delete any non-repeated part of the article."
        " You will keep the inline citations (indicated by [1], [2], etc.) and the article structure (indicated by #, ##, etc.)"
        " appropriately."
        " Do your job for following article."
        "\nDraft Article:\n{draft_article}"
    )

    @classmethod
    def format_prompt(cls, draft_article):
        return {"system": cls.system, "user": cls.user.format(draft_article=draft_article)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class GeneralRAGPPrompt:
    """
    Prompt for the model to provide answers grounded in the provided information.
    The model should ensure that its responses are faithful and accurate based on the context given.
    """

    system = (
        "You are an advanced language model designed to provide accurate and faithful answers."
        " Your responses should be grounded in the information provided to you and should not include any external knowledge."
    )

    user = (
        "Please answer the following question based solely on the information provided. "
        "Ensure that your response is accurate and relevant to the context given. Be as detailed as possible"
        "\nQuestion:\n{question}\n\nContext:\n{context}"
    )

    @classmethod
    def format_prompt(cls, question: str, context: str) -> dict:
        return {"system": cls.system, "user": cls.user.format(question=question, context=context)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)

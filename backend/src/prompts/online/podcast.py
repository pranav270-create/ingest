from typing import Any

import regex
from pydantic import BaseModel, Field, model_validator

from src.prompts.parser import structured_text_cost_parser, text_cost_parser
from src.prompts.base_prompt import BasePrompt
from src.retrieval.retrieve_new import FormattedScoredPoints


def get_category_description(category: str):
    if category == "executive_investor":
        return (
            "This client is a high-achieving professional in a demanding leadership role, likely an"
            " executive or successful investor. They're driven to excel both in their career and personal"
            " life, with a strong focus on maintaining peak mental and physical performance."
            " Key characteristics and needs include:\n"
            "1. Priority on optimizing energy levels and cognitive function for sustained high performance\n"
            "2. Strong interest in cutting-edge health optimization and longevity strategies\n"
            "3. Highly values efficiency and measurable results due to extreme time constraints\n"
            "4. May struggle with work-life balance and the health impacts of a high-stress lifestyle\n"
            "5. Frequently travels, requiring adaptable health and wellness solutions\n"
            "6. Likely has disposable income but places a premium on their time\n"
            "7. May have tried various individual health interventions but lacks a cohesive strategy\n"
            "8. Seeks a comprehensive, science-backed approach that integrates seamlessly into their busy life\n"
            "9. Appreciates high-touch, personalized service that respects their time and status\n"
            "10. Motivated by the concept of extending not just lifespan, but high-quality, high-performance years\n"
            "This client requires a bespoke, time-efficient solution that addresses their unique challenges as a high-powered"
            " individual, promising optimized performance and longevity without compromising their demanding lifestyle."
        )

    elif category == "family_member":
        return (
            "This client is deeply invested in the holistic well-being of their multi-generational family, from young"
            " children to aging parents. They're motivated by a desire to ensure long-term health and vitality for all family"
            " members. Key characteristics and needs include:\n"
            "1. Strong focus on preventative health measures and     longevity for the entire family\n"
            "2. Seeks to instill healthy habits and lifestyle choices across generations\n"
            "3. Values evidence-based approaches but may feel overwhelmed by conflicting health information\n"
            "4. Interested in age-appropriate interventions, from supporting children's development to managing age-related"
            " concerns in older adults\n"
            "5. Likely middle-aged themselves, balancing care for both children and parents\n"
            "6. May struggle with implementing consistent health routines across diverse family needs and preferences\n"
            "7. Seeks guidance on nutrition, supplementation, and wellness practices suitable for different life stages\n"
            "8. Interested in creating a family culture of health and active living\n"
            "9. May have concerns about genetic predispositions and how to mitigate health risks\n"
            "10. Values personalized approaches that consider each family member's unique needs and lifestyle\n"
            "This client requires a comprehensive, family-centric health optimization strategy that addresses the diverse needs"
            " of multiple generations. They need clear, actionable guidance that can be tailored to each family member's age,"
            " health status, and personal goals, fostering a united approach to family wellness and longevity."
        )

    elif category == "concierge_doctor":
        return (
            "This client is a high-end concierge doctor catering to ultra-high-net-worth (UHNW) individuals. They're committed"
            " to providing top-tier healthcare services and are looking to expand their offerings to include comprehensive"
            " healthspan optimization. Key characteristics and needs include:\n"
            "1. Strong desire to elevate their practice by incorporating cutting-edge healthspan services\n"
            "2. Focused on tracking and improving key health drivers for their patients\n"
            "3. Values data-driven approaches and quantifiable results in patient care\n"
            "4. Seeks to offer a more holistic health optimization service without the complexity of managing multiple specialist"
            " teams\n"
            "5. Highly attuned to the unique needs and expectations of UHNW clients\n"
            "6. Interested in partnering with trusted experts to enhance their service offering\n"
            "7. Likely facing time and resource constraints in expanding their practice's capabilities\n"
            "8. Recognizes the growing demand for proactive, longevity-focused healthcare among their clientele\n"
            "9. May have limited expertise in certain areas of health optimization (e.g., advanced nutritional strategies,"
            " cutting-edge fitness protocols)\n"
            "10. Prioritizes maintaining the exclusivity and personalized nature of their concierge service\n"
            "This client requires a turnkey solution that allows them to seamlessly integrate premium healthspan services into"
            " their existing practice. They need a partner who can provide:\n"
            "1. A comprehensive team of health optimization specialists\n"
            "2. Sophisticated systems for tracking and reporting patient progress\n"
            "3. Experience in dealing with the unique demands of UHNW clients\n"
            "4. Seamless integration with their existing concierge medical services\n"
            "5. Ongoing support and education to stay at the forefront of health optimization\n"
            "The ideal solution would allow this concierge doctor to significantly enhance their service offering, improve"
            " patient outcomes, and strengthen their position in the competitive high-end healthcare market, all while"
            " maintaining their focus on providing exceptional, personalized medical care."
        )

    elif category == "gym_owner":
        return (
            "This client is a forward-thinking gym owner looking to elevate their business by integrating premium healthspan"
            " services into their existing fitness offerings. They're motivated to stand out in a competitive market and provide"
            " cutting-edge value to their members. Key characteristics and needs include:\n"
            "1. Ambitious drive to differentiate their gym from standard fitness centers and boutique studios\n"
            "2. Keen interest in expanding into the growing field of healthspan optimization and longevity\n"
            "3. Recognizes the increasing demand for science-backed, comprehensive health services among fitness enthusiasts\n"
            "4. Wants to attract and retain high-value clients by offering more than just traditional workout spaces and"
            " classes\n"
            "5. Lacks the extensive resources, specialized staff, or advanced technology typically associated with dedicated"
            " health optimization centers\n"
            "6. Seeks to bridge the gap between conventional fitness and cutting-edge health science\n"
            "7. May have a solid local reputation but lacks the broader brand recognition of venture-funded health companies\n"
            "8. Interested in data-driven approaches to member health and fitness progress\n"
            "9. Likely operates one or a small chain of gyms, with a hands-on approach to business management\n"
            "10. Values member education and empowerment in health and fitness journeys\n"
            "This gym owner requires a comprehensive solution that allows them to seamlessly integrate premium healthspan"
            " services into their existing gym model. They need a partner who can provide:\n"
            "1. State-of-the-art health assessment tools and protocols\n"
            "2. Expertise in interpreting complex health data and creating personalized optimization plans\n"
            "3. Ongoing training and support for existing staff to confidently deliver new services\n"
            "4. A robust technological backend for tracking member progress and managing health data\n"
            "5. Marketing support to effectively communicate the value of new healthspan services\n"
            "6. A scalable model that can grow with their business\n"
            "The ideal solution would enable this gym owner to transform their business into a cutting-edge health optimization"
            " center, attracting health-conscious clients seeking more than just a place to work out. This upgrade would position"
            " their gym at the forefront of the fitness industry's evolution towards comprehensive health and longevity services,"
            " significantly boosting their competitive edge and member value proposition."
        )
    elif category == "enterprise":
        return (
            "Enterprise clients are large organizations seeking comprehensive healthspan and wellness solutions for their employees."
            " These clients value scalable, data-driven approaches that can be seamlessly integrated into their existing corporate structures."
            " Key characteristics and needs include:\n"
            "1. Focus on employee well-being and productivity\n"
            "2. Interest in preventative health measures and comprehensive wellness programs\n"
            "3. Requires scalable solutions that can cater to a large and diverse workforce\n"
            "4. Values data privacy and secure handling of employee health information\n"
            "5. Seeks measurable outcomes and ROI from wellness initiatives\n"
            "6. Needs integration with existing HR and corporate health systems\n"
            "7. Interested in personalized wellness plans for employees while maintaining overall program consistency\n"
            "8. May have existing wellness programs but looks to enhance them with advanced healthspan strategies\n"
            "9. Values partnerships with trusted wellness providers that offer comprehensive support\n"
            "10. Requires solutions that can adapt to various geographies and cultural contexts within the organization\n"
            "This enterprise client requires a robust, scalable healthspan optimization solution that can be seamlessly integrated into"
            " their existing corporate wellness programs. They need a partner who can provide:\n"
            "1. Comprehensive wellness programs tailored to diverse employee needs\n"
            "2. Advanced data analytics tools to track and measure program effectiveness\n"
            "3. Secure and compliant handling of employee health data\n"
            "4. Flexible integration with existing corporate systems and workflows\n"
            "5. Ongoing support and training for HR and wellness coordinators\n"
            "6. Customizable solutions that can adapt to different departments and geographies\n"
            "The ideal solution would empower this enterprise to enhance their employee wellness initiatives, improve overall productivity,"
            " and foster a healthier, more engaged workforce through cutting-edge healthspan optimization strategies."
        )
    else:
        return ""
    

class SummarizeChunkClaimsPrompt(BasePrompt):
    """
    This prompt summarizes the claims made in a chunk of podcast transcript.
    """

    system = (
        "You are an expert in summarizing scientific and health-related information. "
        "Your task is to extract and summarize the key claims made in a portion of a podcast transcript."
    )

    user = (
        "You will be given a chunk of text from a health and science podcast transcript. "
        "Your job is to summarize the main claims or points made in this chunk. "
        "Focus on scientific or health-related claims, and be concise but accurate. "
        "Chunk of transcript:\n{chunk}"
    )

    @classmethod
    def format_prompt(cls, chunk):
        return {"system": cls.system, "user": cls.user.format(chunk=chunk)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class ExtractPodcastChunkClaims(BasePrompt):
    """
    This prompt uses structured output to extract the key scientific claims from a podcast transcript.
    """

    system = (
        "You are an expert in biology, medicine, and health. Your task is to extract key scientific claims from the podcast"
        " transcript. Focus on identifying specific claims related to science, medicine, and personal health, and provide"
        " context for each claim."
    )

    user = (
        "You will be given a section of a transcript of a podcast on health and science."
        " Your job is to extract the key, specific scientific claims made in the section."
        " Be precise and detailed. Please output the text in the provided structure."
        " Podcast Transcript Section:\n{chunk}"
    )

    class DataModel(BaseModel):
        """
        DataModel for extracting thorough, scientifically rigorous information from a podcast transcript
        """

        podcast_clip_claims: list[str] = Field(..., description="A list of the key claims made in the podcast")

    @classmethod
    def format_prompt(cls, chunk):
        return {"system": cls.system, "user": cls.user.format(chunk=chunk)}

    @staticmethod
    def parse_response(response: DataModel, model: str) -> tuple[list[str], list[str], float]:
        text, cost = structured_text_cost_parser(response, model)
        return text.podcast_clip_claims, cost


class LabelClustersPrompt(BasePrompt):
    """
    This prompt labels the clusters of claims based on their relevance to a given topic.
    """

    system = "You are an expert in biology, medicine, and health."

    user = (
        "You are provided a set of claims made from a podcast in which {host_name} interviews {guest_name}."
        " Your task is to analyze how the claims relate to the Topic of conversation and categorize their relevance."
        "\nTopic:\n{topic}"
        "\nClaims:\n{claims}"
    )

    class DataModel(BaseModel):
        description: str = Field(
            ...,
            description="A summary describing the subtopic reflected in the theme of the claims in relation to the Topic.",
        )
        subtopic: str = Field(
            ..., description="A title for the group of claims that represents a meaningful subtopic of the Topic."
        )

    @classmethod
    def format_prompt(cls, claims, host_name, guest_name, topic):
        claims = "\n".join([f"{i+1}. {claim['claim']}" for i, claim in enumerate(claims)])
        return {
            "system": cls.system,
            "user": cls.user.format(claims=claims, host_name=host_name, guest_name=guest_name, topic=topic),
        }

    @staticmethod
    def parse_response(response: DataModel, model: str) -> tuple[dict[str, str], float]:
        text, cost = structured_text_cost_parser(response, model)
        return {"description": text.description, "subtopic": text.subtopic}, cost


class WriterPromptPodcast:
    """
    Prompt for the WriterAgent to generate questions based on the claim and perspective.
    """

    system = "You know about health and wellness."

    user = (
        "You meet the following client description and listen to popular health and wellness podcasts. You heard a claim that was"
        " made in a recent podcast and have an opportunity to ask a question to a health and wellness expert."
        " Given the claim, pose an interesting, insightful question to the expert on the topic to uncover a deeper understanding"
        " of the truth."
        "\nClient Description:\n{client_profile}"
        "\nClaim:\n{claim}\n"
        "{history}\n"
        "Question:\n"
    )

    @classmethod
    def format_prompt(cls, claim, client_profile, history):
        if len(history) < 100:
            history = ""
        return {"system": cls.system, "user": cls.user.format(claim=claim, client_profile=client_profile, history=history)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class ExpertPromptPodcast:
    """
    Prompt for the ExpertAgent to answer questions by breaking them into sub-questions and retrieving context.
    """

    system = "You are an expert at biology, medicine, and health. You are scientifically rigorous."

    entry_template = "Title: {title}\nContext: {context}\n" "Relevance Score: {score}\n"

    user_template = (
        "\nEvidence:\n{evidence}"
        "Original Context:\n{raw_string}\n"
        "You are a scientific expert responding to a claim that was made on a popular health podcast."
        " Using the evidence and original context, respond to the claim. Valid responses include debunking the claim when it is contrary"
        " to the evidence, or explaining where the claim is limited in scope or applicability."
        " Only make arguments supported by evidence. Be thorough and descriptive in your analysis."
        "\nClaim:\n{claim}"
    )

    @classmethod
    def format_prompt(cls, claim, entries: list[FormattedScoredPoints], raw_string: str):
        formatted_entries = "\n".join(
            [cls.entry_template.format(title=e.title, context=e.raw_text, score=e.score) for e in entries]
        )
        return {
            "system": cls.system,
            "user": cls.user_template.format(
                claim=claim,
                evidence=formatted_entries,
                raw_string=raw_string
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> str:
        response, _ = text_cost_parser(response, model)
        return response


class EmailPromptPodcast:
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
        "1. Podcast Reference (1 sentence):"
        "\nHost: {host_name}\nTopic: {topic}\nKey Claims: {claims}\n"
        "2. Validate and Expand (2-3 sentences):"
        " Given evidence, validate the claim and explain its relevance to the client's health goals."
        " Utilize the Expert Opinion to validate, debunk, or expand on the claims.\n"
        "3. Bridge to Services (2-3 sentences):"
        "Highlight how your company's approach addresses the family's diverse needs and enhances the podcast advice.\n"
        "Expert Opinions:\n"
        "{responses}\n"
        "Tone: Knowledgeable, personalized, and subtly persuasive. Don't be overly salesy. Don't write a call-to-action."
        " Tailor the message to address client profile.\n"
        "Maximum length: 300 words\n"
        "Note: Avoid typical email greetings or signatures. Focus on creating a concise message that resonates with"
        " the client profile. Do not make health recommendations and do not directly try to sell."
        "EXAMPLES:\n"
        "Here is an example of the kind of note we're looking for. Notice how it is short and to the point and do not try"
        " to sell the individual directly. Use these as inspiration for style and structure, but create a unique message"
        " tailored to the given client profile and podcast information:\n"
        "Example 1:\n"
        "---\n"
        "Not sure if you caught the Peter Attia Drive or Huberman podcasts last week but a few really interesting points for"
        " partners related to our initial conversation. Huberman had Dr. Charan Raganath on to talk about cognition."
        " Dr. Ragantath said many of the most important factors for **maintaining memory as we age** are related to general"
        " health: Sleep, exercise, social engagement, avoiding smoking, most food from unprocessed or minimally processed "
        " maintaining a sense of purpose. Our POV: We 100% agree, as you might expect - and hold our members accountable to"
        " changes in these arenas. As you mentioned, it's one thing to hear it on a podcast -- it's quite another to have a"
        " personal team making it accessible and keeping each exec on track."
        "---\n"
        # "Example 2:\n"
        # "---\n"
        # "Attia had Dr. Marty Makary on to talk about the medical system generally BUT they touched on the large preventative"
        # " impact of Hormone Replacement Therapy (HRT) for postmenopausal women. While we don't perform HRT, we do refer out"
        # " in cases where relevant. This is the advantage of having a personal 'Chief Health Officer' working across disciplines"
        # " keeping up with the trends, for all genders."
        # "---\n"
        "Now, using the provided client profile and podcast information, create a new, unique note following a similar style and"
        " structure:\n"
    )

    @classmethod
    def format_prompt(cls, host_name: str, topic: str, client_profile: str, claims: list[str], responses: list[str]):
        formatted_responses = "\n".join(
            [f"Claim {i+1}: {c}\nResponse {i+1}: {r}" for i, (c, r) in enumerate(zip(claims, responses))]
        )  # noqa
        return {
            "system": cls.system,
            "user": cls.user.format(
                host_name=host_name,
                topic=topic,
                client_profile=client_profile,
                claims=claims,
                responses=formatted_responses,
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> str:
        response, _ = text_cost_parser(response, model)
        return response


class Fact(BaseModel):
    """
    Class representing single statement.
    Each fact has a body and a list of sources.
    If there are multiple facts make sure to break them apart such that each one only uses a set of sources
    that are relevant to it.
    """

    fact: str = Field(
        ...,
        description="Body of the sentences, as part of a response, it should read like a sentence that answers the question",
    )
    substring_quotes: list[str] = Field(
        ...,
        description="Each source should be a direct quote from the context, as a substring of the original content",
    )

    @model_validator(mode="after")
    def validate_sources(self, info) -> "Fact":
        if info.context is None:
            return self

        formatted_context = info.context.get("formatted_context", "")
        valid_quotes = []

        for quote in self.substring_quotes:
            if self._find_span(quote, formatted_context):
                valid_quotes.append(quote)

        self.substring_quotes = valid_quotes
        return self

    def _find_span(self, quote, context, max_errors=5):
        for errors in range(max_errors + 1):
            pattern = f"({quote}){{e<={errors}}}"
            match = regex.search(pattern, context)
            if match:
                return True
        return False


class FactCheckPromptPodcast:
    """
    Prompt for the ExpertAgent to answer questions by breaking them into sub-questions and retrieving context.
    """

    system = "You are an expert at biology, medicine, and health. You are scientifically rigorous."

    entry_template = "Context: {context}\n" "Relevance Score: {score}\n"

    user_template = (
        "\nClaim:\n{claim}"
        "\nEvidence:\n{evidence}"
        "You are fact checking a claim that was made on a popular health and science podcast."
        "Tip: Make sure to cite your sources, and use the exact words from the context."
    )

    class DataModel(BaseModel):
        """
        Class representing a claim and its fact-check as a list of facts each one should have a source.
        Each sentence contains a body and a list of sources.
        """

        claim: str = Field(
            ...,
            description="Claim that was made",
        )

        fact_check: list[Fact] = Field(
            ..., description="Body of the answer, each fact should be its separate object with a body and a list of sources"
        )

        @model_validator(mode="after")
        def validate_facts(self):
            self.fact_check = [fact for fact in self.fact_check if fact.substring_quotes]
            return self

    @classmethod
    def format_prompt(cls, claim, entries: list[dict[str, Any]]):
        formatted_entries = "\n".join(
            [f'Evidence {i+1}:\n{cls.entry_template.format(context=e["input"], score=e["score"])}' for i, e in enumerate(entries)]
        )
        return {
            "system": cls.system,
            "user": cls.user_template.format(claim=claim, evidence=formatted_entries),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[DataModel, float]:
        response_model, cost = structured_text_cost_parser(response, model)
        return response_model, cost


class TextPromptPodcast:
    """
    Prompt for the Agent to generate a personalized text message for a prospective client based on a recent popular health podcast.
    """

    system = (
        "You are a health optimization expert at Apeiron Life, specializing in personalized communication for high-end clients."
        " You have deep knowledge of health podcasts, current wellness trends, and the needs of busy professionals."
    )

    user = (
        "Write a concise, personalized text message to a client with the following profile:"
        "{client_profile}"
        "Reference a recent podcast episode using this structure:\n"
        "1. Podcast Reference (1 sentence):"
        "\nHost: {host_name}\nTopic: {topic}\nKey Claims: {claims}\n"
        "2. Validate and Expand (1-2 sentences):"
        " Given evidence, validate the claim and explain its relevance to the client's health goals."
        " Utilize the Expert Opinion to validate, debunk, or expand on the claims.\n"
        "3. Bridge to Services (1-2 sentences):"
        "Highlight how your company's approach addresses the client's needs and enhances the podcast advice.\n"
        "Expert Opinions:\n"
        "{responses}\n"
        "Tone: Friendly, concise, and engaging. Do not include a call-to-action."
        "Maximum length: 200 words\n"
        "Note: Focus on creating a brief message that resonates with the client profile without making direct health recommendations or sales pitches."
    )

    @classmethod
    def format_prompt(cls, host_name: str, topic: str, client_profile: str, claims: list[str], responses: list[str]):
        formatted_responses = "\n".join(
            [f"Claim {i+1}: {c}\nResponse {i+1}: {r}" for i, (c, r) in enumerate(zip(claims, responses))]
        )
        return {
            "system": cls.system,
            "user": cls.user.format(
                host_name=host_name,
                topic=topic,
                client_profile=client_profile,
                claims=claims,
                responses=formatted_responses,
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> str:
        response, _ = text_cost_parser(response, model)
        return response


class BlogPromptPodcast:
    """
    Prompt for the Agent to generate a blog post based on a recent popular health podcast.
    """

    system = (
        "You are a health optimization expert at Apeiron Life, specializing in creating detailed and informative blog posts."
        " You have extensive knowledge of health podcasts, wellness trends, and the needs of a professional audience."
    )

    user = (
        "Write a comprehensive blog post for a professional audience with the following details:"
        "{client_profile}"
        "Reference a recent podcast episode using this structure:\n"
        "1. Introduction (2-3 sentences):"
        "Introduce the podcast episode, mentioning the host, guest, and main topic.\n"
        "2. Key Claims (3-5 paragraphs):"
        "Discuss each key claim made in the podcast, providing validation, debunking, or expansion based on expert opinions."
        " Utilize the provided evidence to support your analysis.\n"
        "3. Implications (2 paragraphs):"
        "Explain the relevance of these claims to the reader's health and wellness goals.\n"
        "4. Conclusion (2-3 sentences):"
        "Summarize the key takeaways and how Apeiron Life's services can further support the reader's health optimization journey.\n"
        "Expert Opinions:\n"
        "{responses}\n"
        "Tone: Informative, authoritative, and engaging. Aim for clarity and depth without being overly technical."
        "Maximum length: 1500 words\n"
        "Note: Focus on creating insightful and valuable content that educates the reader and establishes Apeiron Life as a thought leader."
    )

    @classmethod
    def format_prompt(cls, host_name: str, topic: str, client_profile: str, claims: list[str], responses: list[str]):
        formatted_responses = "\n".join(
            [f"Claim {i+1}: {c}\nResponse {i+1}: {r}" for i, (c, r) in enumerate(zip(claims, responses))]
        )
        return {
            "system": cls.system,
            "user": cls.user.format(
                host_name=host_name,
                topic=topic,
                client_profile=client_profile,
                claims=claims,
                responses=formatted_responses,
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> str:
        response, _ = text_cost_parser(response, model)
        return response


class TweetPromptPodcast:
    """
    Prompt for the Agent to generate a tweet based on a recent popular health podcast.
    """

    system = (
        "You are a health optimization expert at Apeiron Life, specializing in creating concise and impactful social media content."
        " You have a keen understanding of health podcasts, wellness trends, and the needs of an online audience."
    )

    user = (
        "Write a tweet based on the following podcast episode details:"
        "{client_profile}"
        "Use the structure below to ensure clarity and engagement:\n"
        "1. Key Point (280 characters):"
        "Summarize the most impactful claim or insight from the podcast episode.\n"
        "Include relevant hashtags and mentions where appropriate.\n"
        "2. Call to Engagement (optional, within the character limit):"
        "Encourage followers to listen to the podcast or share their thoughts.\n"
        "Tone: Concise, engaging, and informative. Utilize popular hashtags related to health and wellness."
        "Maximum length: 280 characters\n"
        "Note: Focus on creating a tweet that captures the essence of the podcast episode while prompting audience interaction."
    )

    @classmethod
    def format_prompt(cls, host_name: str, topic: str, client_profile: str, claims: list[str], responses: list[str]):
        # Typically, a tweet focuses on one key claim. We'll take the first claim for this purpose.
        if claims:
            claim = claims[0]
            response = responses[0] if responses else ""
        else:
            claim = ""
            response = ""
        return {
            "system": cls.system,
            "user": cls.user.format(
                host_name=host_name,
                topic=topic,
                client_profile=client_profile,
                claims=[claim],
                responses=[response],
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> str:
        response, _ = text_cost_parser(response, model)
        return response


class GeneralFollowUpPrompt:
    """
    Prompt for the Agent to modify existing content based on a follow-up instruction while maintaining the original structure.
    """

    system = (
        "You are a health optimization expert at Apeiron Life, specializing in personalized communication for high-end clients."
        " You have deep knowledge of health podcasts, current wellness trends, and the needs of busy professionals seeking to"
        " improve their health and performance. Your task is to refine and modify existing content while maintaining its core"
        " message and professional tone."
    )

    user = (
        "Review and modify the following content based on the follow-up instruction, while maintaining the original structure"
        " and using the provided context:\n\n"
        "Original Content:\n{original_content}\n\n"
        "Follow-Up Instruction:\n{follow_up_prompt}\n\n"
        "Available Context:\n"
        "Claims: {claims}\n"
        "Citations: {citations}\n\n"
        "Guidelines:\n"
        "1. Maintain the original structure and professional tone\n"
        "2. Incorporate the requested changes from the follow-up instruction\n"
        "3. Continue to reference the provided claims and citations appropriately\n"
        "4. Keep the content focused on health optimization and wellness\n"
        "5. Avoid making direct health recommendations or being overly salesy\n"
        "6. Ensure the modified content aligns with Apeiron Life's expertise and services\n\n"
        "Please provide the modified version while preserving the core message and professional quality:"
    )

    @classmethod
    def format_prompt(cls, original_content: str, follow_up_prompt: str, citations: list[list[dict]], claims: list[str]):
        formatted_citations = "\n".join([
            f"Citation Group {i+1}: " + ", ".join([c.get("title", "Untitled") for c in group])
            for i, group in enumerate(citations)
        ])

        return {
            "system": cls.system,
            "user": cls.user.format(
                original_content=original_content,
                follow_up_prompt=follow_up_prompt,
                claims=claims,
                citations=formatted_citations
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> str:
        response, _ = text_cost_parser(response, model)
        return response


class SummarizePodcast:
    """
    Prompt for generating a comprehensive summary of a podcast based on clustered claims and subtopics.
    """

    system = (
        "You are an expert at biology, medicine, and health with a talent for synthesizing complex information "
        "into clear, engaging summaries. You are scientifically rigorous while maintaining accessibility."
    )

    user_template = (
        "You will be provided with key segments from a health and science podcast, organized by subtopics. "
        "Create a comprehensive yet concise summary of the podcast discussion using the following exact format:\n\n"
        "**Overall Summary**\n"
        "[2-3 sentences introducing the main themes and significance]\n\n"
        "[For each subtopic:]\n"
        "**Subtopic Title**\n"
        "**Description**\n"
        "**Points**\n"
        "- [Key point 1]\n"
        "- [Key point 2]\n"
        "...\n\n"
        "**Takeaways**\n"
        "[3-4 bullet points highlighting the most important conclusions]\n\n"
        "Podcast Segments:\n{formatted_clusters}\n\n"
        "Guidelines:\n"
        "1. Strictly follow the formatting with the exact headers shown above\n"
        "2. Maintain scientific accuracy while being accessible\n"
        "3. Include all relevant subtopics from the provided segments\n"
        "4. Ensure each section is concise but comprehensive\n"
        "5. Use bullet points for Points and Takeaways sections\n\n"
        "Please provide the summary following this exact structure."
    )

    cluster_template = (
        "Subtopic: {subtopic}\n"
        "Description: {description}\n"
        "Key Claims:\n{claims}\n"
        "---\n"
    )

    @classmethod
    def format_prompt(cls, formatted_clusters: list[dict]) -> dict:
        """
        Format the prompt with clustered podcast data.
        
        Args:
            formatted_clusters: List of dictionaries containing subtopics, descriptions, 
                              claims, timestamps, and raw strings
        """
        formatted_sections = []
        for cluster in formatted_clusters:
            claims_text = "\n".join([f"- {claim}" for claim in cluster["claims"]])
            formatted_sections.append(
                cls.cluster_template.format(
                    subtopic=cluster["subtopic"],
                    description=cluster["description"],
                    claims=claims_text
                )
            )
        
        return {
            "system": cls.system,
            "user": cls.user_template.format(
                formatted_clusters="\n".join(formatted_sections)
            )
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> str:
        """Parse the response and return the summary."""
        response, _ = text_cost_parser(response, model)
        return response

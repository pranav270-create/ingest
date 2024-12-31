import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parents[2]))

from src.prompts.parser import text_cost_parser
from src.prompts.registry import PromptRegistry
from src.schemas.schemas import Entry


@PromptRegistry.register("client_qa")
class ClientQAPrompt:
    system = (
        "You are an advanced AI assistant integrated into a healthcare application. Your role is to provide scientifically grounded, personalized health recommendations and insights. You have access to the client's historical data, including past interactions with the healthcare team, medical history, and lifestyle information. Your responses should:"  # noqa
        "\n1. Be based solely on the provided context and scientific evidence."
        "\n2. Maintain consistency with the tone, style, and content of previous interactions between the client and the healthcare team."  # noqa
        "\n3. Prioritize patient safety and adhere to medical best practices."
        "\n4. Avoid introducing external information not present in the given context."
        "\n5. Use clear, professional language that is accessible to the client."
        "\n6. Provide explanations for your recommendations, referencing relevant data points from the client's history."
        "\nRemember, your goal is to support and enhance the care provided by the human healthcare team, not to replace it."
    )

    user = (
        # "\n\nClient Profile:\n{profile}"
        "\n\nClient Question:\n{question}"
        "\n\nHistorical Context:\n{context}"
        "Based on the following client profile, client question and historical context, provide a health recommendation that aligns with the healthcare team's approach and is grounded in scientific evidence. Ensure your response maintains the established tone and style of communication."  # noqa
        "Ensure to answer the client question specifically. This is the most important"
        "\n\nPlease provide your recommendation, explaining your reasoning and referencing relevant aspects of the client's history. Include any necessary caveats or suggestions for follow-up with the healthcare team. For context, today is {date}. Sign off, 'Your Healthcare Team'."  # noqa
    )

    @classmethod
    def format_prompt(cls, question: str, context: list[dict], profile_text: str, today_date: str, top_k: int) -> dict:
        dummy = profile_text  # noqa
        sorted_context = context[:top_k]
        sorted_context = sorted(context, key=lambda x: x["date"])
        # Format context items
        formatted_context = []
        for item in sorted_context:
            document_title = item["title"]
            content = item["content"]
            date = item["date"]
            formatted_item = f"[{date}] {document_title.upper()}:\n{content}\n"
            formatted_context.append(formatted_item)

        # Join formatted context items
        context_str = "\n".join(formatted_context)
        return {"system": cls.system, "user": cls.user.format(question=question, context=context_str, date=today_date)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = text_cost_parser(response, model)
        # Remove any signature or closing remarks
        response = re.sub(
            r"\s*(?:Best|Sincerely|Regards|Yours truly|Thank you|Warm regards|Kind regards|Yours sincerely|Respectfully|Cordially|Cheers|Yours faithfully|Yours respectfully|Yours|Faithfully|Respectfully yours|All the best|Take care|Yours truly|Yours very truly|Very truly yours|With appreciation|With gratitude|With thanks|With best wishes|With best regards|With warm regards|With sincere thanks|With sincere appreciation|Yours gratefully|Gratefully|Gratefully yours|Appreciatively|Appreciatively yours|Sincerely yours|Yours very sincerely|Very sincerely yours|Warmly|Warmest regards|Warmest wishes|Best wishes|Best regards|Kindest regards|Kindly|Yours kindly|Yours very kindly|Very kindly yours|Yours respectfully|Respectfully yours|Yours obediently|Obediently yours|Yours very respectfully|Very respectfully yours|Yours most sincerely|Most sincerely yours|Yours most respectfully|Most respectfully yours|Yours most faithfully|Most faithfully yours|Yours most gratefully|Most gratefully yours|Yours most appreciatively|Most appreciatively yours|Yours most obediently|Most obediently yours|Yours most truly|Most truly yours|Yours most kindly|Most kindly yours|Yours most cordially|Most cordially yours|Yours most warmly|Most warmly yours|Yours most affectionately|Most affectionately yours|Yours most devotedly|Most devotedly yours|Yours most humbly|Most humbly yours|Yours most respectfully and sincerely|Stay well|Most respectfully and sincerely yours|Your [A-Za-z\s]+ AI Assistant),?\s*(?:\[.*?\])?\s*$",  # noqa
            "",
            response,
            flags=re.IGNORECASE,
        ).strip()
        return response, cost


@PromptRegistry.register("client_trends")
class ClientTrendsPrompt:
    system = (
        "You are an advanced AI assistant specializing in wearable health data analysis and personalized recommendations. Your role is to interpret weekly health statistics and provide scientifically grounded insights and recommendations. You have access to the client's current wearable data and historical context of similar data patterns with expert recommendations. Your analysis should focus on the following key metrics:"  # noqa
        "\n- sleep_avg_hrv: Average Heart Rate Variability during sleep, per night in the week, a key indicator of recovery and overall health."  # noqa
        "\n- total_hours_asleep: Average hours asleep (in hours) per night in the week, crucial for assessing sleep quantity."
        "\n- resistance_duration: Cumulative time spent on resistance training (in hours) across the week, important for muscle health and metabolism."  # noqa
        "\n- activity_duration: Cumulative total duration of physical activity (in hours) across the week, a measure of overall movement and exercise."  # noqa
        "\n- all_hrz_1 to all_hrz_5: Cumulative time spent (in hours) across the week in different heart rate zones, from low intensity (1) to high intensity (5), indicating exercise intensity distribution."  # noqa
        "\nThese metrics provide a comprehensive view of the client's sleep quality, physical activity, and exercise intensity. Your recommendations should:"  # noqa
        "\n1. Be based solely on the provided wearable data and historical context."
        "\n2. Align with previous expert recommendations for similar data patterns."
        "\n3. Prioritize the client's health and safety, adhering to established medical and fitness guidelines."
        "\n4. Provide clear, actionable advice that is accessible to the client."
        "\n5. Explain the reasoning behind each recommendation, referencing specific data points."
        "\n6. Suggest areas for improvement or maintenance based on the data trends."
        "\n7. Highlight any significant changes or patterns compared to historical data."
        "\nYour goal is to support and enhance the care provided by the human healthcare team by offering data-driven insights and recommendations."  # noqa
    )

    user = (
        # "\n\nClient Profile:\n{profile}"
        "\n\nHistorical Context (Similar patterns with expert communications):\n{historical_context}"
        "Based on the following weekly wearable health statistics and historical context of similar data patterns with expert recommendations, provide a comprehensive health analysis and personalized recommendations."  # noqa
        "\n\nCurrent Week's Health Statistics:\n{current_stats}"
        "\n\nPlease provide your analysis and recommendations, explaining your reasoning and referencing both the current week's data and relevant historical patterns. Please copy the style of past recommendations, and include insights on sleep and exercise only."  # noqa
        "For context, today is {date}."
    )

    @classmethod
    def format_prompt(cls, current_stats: dict, historical_context: list[dict], profile_text: str, end_date: str) -> dict:
        dummy = profile_text  # noqa
        current_stats_str = "\n".join([f"{k}: {v}" for k, v in current_stats.items()])
        context_str = "\n\n".join(
            [
                f"Similar Pattern {i+1}:\nStats: {pattern['stats']}\nExpert Communications: {pattern['recommendation']}"
                for i, pattern in enumerate(historical_context)
            ]
        )
        return {
            "system": cls.system,
            "user": cls.user.format(current_stats=current_stats_str, historical_context=context_str, date=end_date),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


@PromptRegistry.register("summary_email")
class SummaryEmailPrompt:
    system = "You are an advanced AI assistant specializing in wearable health data analysis and personalized recommendations. Your role is to interpret weekly health statistics and provide scientifically grounded insights and recommendations. You have access to the client's current wearable data and historical context of similar data patterns with expert recommendations."  # noqa

    user = (
        "Here are past weekly summaries emails for your reference:\n{weekly_summaries}"
        "Here is the summary of the week: {week_summary}"
        "Based on the following weekly wearable summary, please draft a Summary Email like in the past, summarizing what happened this week and specific recommendations for Exercise and Sleep, based on the wearable data. Please copy the style of past emails."  # noqa
    )

    @classmethod
    def format_prompt(cls, weekly_summaries: str, week_summary: str) -> dict:
        return {"system": cls.system, "user": cls.user.format(weekly_summaries=weekly_summaries, week_summary=week_summary)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class BasicSummaryEmailPrompt:
    system = "You are an advanced AI assistant specializing in wearable health data analysis and personalized recommendations. "

    user = """
        You and a team of experts coach a client to optimize their health through exercise, sleep, and nutrition.
        Your role is to summarize their week's data in three pieces, exercise, sleep, and nutrition.
        We collected the following data:
        Text Messages - messages between the client and the team. These messages often provide context on the client's life and show what they are interested in.
        Internal Notes - notes that the team sends to each other about a client. These notes provide important context on the client's life.
        Using the information, please draft a summary email to send to the client to note what happened to their sleep, exercise, and nutrition, where they
        succeeded, how they failed, and what changes might need to be made to help them succeed.
        Example:
        Text Messages:\n
        2022-03-28 16:47:35 - Staff 185: Hi Geoff, here in the chat.  Click the "picture" icon with the mountains to attach a photo.  \n\nTravel days, home days, they all count so no problem on what day this happens to fall on.\n'
        2022-04-03 17:28:36 - Staff 277: Master's program is on App\n
        Internal Notes:
        Trends:
        Summary Email: Overview:\n- Geoff!\n\nProgram sent and on app, daily stretches sent via text and sleep/nutrition insights below.\n\nA unique week as you will be on your feet daily walking the course, long days and topped off with late night dinners. Be smart and follow insights below!\n
        -Rejuvenation Recommendation: A solid 2 nights of 6+ hours of sleep.  Well done!  With the other nights sub-6 and a decrease in activity last week, your HRV took a dip.  Though still quite high compared to many other clients.  Let's get yours back up to what we know you're capable of though.\n\nThe aim: 6-6.5 hours each night instead of the 5-6 hour range at times.  Your bed time has been the most effective way for you to improve duration in the past. \n\n- Have you experimented with having a little less protein?  Notice any difference?\n- Glycine can go up to 3 grams a night\n- Magnesium glycinate - 400-600 mg a night\n- PS - follow label. You can take PS with Magnesium and Glycine. If you have more questions on your sleep supplements, let me know.
        -Exercise Recommendation: Solid week. Albeit I notice that your total time is down quite a bit. This could be due to less gold as your activity was down almost 4 hours compared to the last 3 weeks. And your specific aerobic work is also down by half, = to 60 minutes. Resistance training is up which is great.\n\nAs noted below, I would like to see a minimum 3x a week minimum 30 minutes of cardio. Approximately 60 minutes in Z3 and 30 in Z4/5. This does not include cardio that you get during your resistance sessions. These are specific to cycling, treadmill walking protocol or we can design some other cardio workouts which take advantage of the rower and skier. Options a plenty! \n\nThis combined with the resistance, movement quality and activity should help you to easily accomplish the HR ideal and get appropriate response in each of the HR zones.\n\nNote: this is a great week to ramp up and build in the discipline and consistency for the cardio and keep up with the 3 days of strength and movement quality.
        -Nutrition Recommendation: Nice work moderating your alcohol this week!  Not only many days with zero drinks, but on the days you did drink it was quite moderate.  This is very helpful as we aim to modulate your weight and body composition.\n\nHow are your balanced plates going? As discussed, the goal of a balanced plate is to ensure you're getting all the nutrients you need in each bite.\n\nRemember:\n\n- Healthy fat\n- Protein - I'm getting the protein resources for you - will send soon. \n- High quality carb (veggie, fruit, starch, or a mix)\n- Nutrition Chart URL: None\n- Sleep Chart URL: None\n\nRejuvenation:\n- Recommendation: Some rough nights last week, and you're headed into Augusta. HRV went way up though! So you're doing something right, even though duration was low.\n\nYour main \"go-tos\" to support sleep:\n- Watch alcohol\n- Avoid eating too late in social events\n- Give yourself enough time in bed\n- Use a recovery tool daily
        -Exercise:\n- Workout Recommendation: We need more cardio and consistency in this area. Less willy nilly and more program design and support. We need discipline and we can easily work this in and put together a plan you are willing to execute, giving you some autonomy but with guidelines as stated above. \n\nThe resistance program is strong! Solid work this week on resistance training. 3 days a week is our goal and this is generally what you accomplish in terms of a weeks worth of workouts. Nice amount of time allotted to aerobic work with a majority dedicated to cycling. More to discuss here....\n
        \n-Heart Recommendation: Nice week surpassing your ideal and 50 minutes at Z4/5. Great balance of Z3 and Z4/5 work = to approximately 150 minutes. This is perfect in terms of total time spent and spread across these 3 zones. Let's build off this and aim for 3 days of specific cardio sessions i.e. bike, hike, treadmill walking protocol, rowing or skiing. \n\nFor Z4/5 workouts focus on rowing skiing and treadmill walking protocol\nFor Z2/3 steady state (and Z4/5) focus on cycling\n
        Follow the style and structure, but use the data for this week
        Text Messages:\n{text_messages}\n
        Internal Notes:\n{internal_notes}\n
        Trends:\n{trends}\n
        Summary Email:\n
"""  # noqa

    @classmethod
    def format_prompt(cls, weekly_summaries: str, week_summary: str) -> dict:
        return {"system": cls.system, "user": cls.user.format(weekly_summaries=weekly_summaries, week_summary=week_summary)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class UnstructuredFeatureExtractionPrompt:
    system = "You extract and categorize health-related information from coaching communications."

    user = """Extract key health information from the provided coaching communications.
    Include only explicit advice and mentions, not interpretations. Only include one item per tag.

Input format:
<text_messages>Client-coach messages</text_messages>
<internal_notes>Staff internal notes</internal_notes>
<summary_email>Weekly health summary and recommendations</summary_email>

Required output format:
<exercise_advices>
    <exercise_advice>One specific exercise recommendation per tag</exercise_advice>
</exercise_advices>

<sleep_advices>
    <sleep_advice>One specific recommendation to improve sleep per tag</sleep_advice>
</sleep_advices>

<nutrition_advices>
    <nutrition_advice>One specific recommendation to improve nutrition per tag</nutrition_advice>
</nutrition_advices>

<travel_plans>
    <travel_plan>One specific travel plan per tag</travel_plan>
</travel_plans>

<injuries>
    <injury>One specific injury mention per tag</injury>
</injuries>

<goals>
    <goal>One specific goal mention per tag</goal>
</goals>

<todos>
    <todo>One specific todo mention per tag</todo>
</todos>

<text_messages>{text_messages}</text_messages>
<internal_notes>{internal_notes}</internal_notes>
<summary_email>{summary_email}</summary_email>

"""

    @classmethod
    def format_prompt(cls, text_messages: list[str], internal_notes: list[str], summary_email: str) -> dict:
        text_messages_str = "\n".join(text_messages)
        internal_notes_str = "\n".join(internal_notes)
        return {
            "system": cls.system,
            "user": cls.user.format(
                text_messages=text_messages_str, internal_notes=internal_notes_str, summary_email=summary_email
            ),
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class SummaryEmailFeatureExtraction:
    system = """You are a precise health information extraction system. Your role is to identify and extract all specific recommendations and advice from coaching communications, maintaining the exact language used while avoiding duplicate information."""  # noqa

    user = """Extract all health-related recommendations from the provided text into clear, categorized lists.
Rules:
- Extract only explicit recommendations or advice (not observations or status updates)
- Include quantitative targets when provided
- Keep the original wording where possible
- Each item should be a single, specific recommendation
- Do not include duplicate information
- Do not include URLs or references to charts

Input format:
<text>
{text}
</text>

Required output format:
<health_recommendations>
    <exercise_recommendations>
        <recommendation>Single specific exercise recommendation</recommendation>
        <recommendation>Another exercise recommendation</recommendation>
    </exercise_recommendations>

    <sleep_recommendations>
        <recommendation>Single specific sleep recommendation</recommendation>
        <recommendation>Another sleep recommendation</recommendation>
    </sleep_recommendations>

    <nutrition_recommendations>
        <recommendation>Single specific nutrition recommendation</recommendation>
        <recommendation>Another nutrition recommendation</recommendation>
    </nutrition_recommendations>
</health_recommendations>
"""  # noqa
    assistant = "<health_recommendations>"

    @classmethod
    def format_prompt(cls, summary_email: str) -> dict:
        return {"system": cls.system, "user": cls.user.format(text=summary_email), "assistant": cls.assistant}

    @classmethod
    def parse_response(cls, response: Any, model: str) -> tuple[str, float]:
        response, cost = text_cost_parser(response, model)
        response = cls.assistant + response
        return response, cost


class MessageAndNoteExtractionPrompt:
    system = "You extract and categorize current health status and travel information from coaching communications, with special focus on any conditions that could affect exercise recommendations."  # noqa

    user = """Analyze the provided communications and extract:
1. ALL current physical/health conditions that could affect exercise or nutrition recommendations
2. CURRENT travel status only (not future plans)

Input format:
<text_messages>Client-coach messages</text_messages>
<internal_notes>Staff internal notes</internal_notes>

Required output format:
<current_travel_status>
    <thinking>Think about whether the client is currently traveling, talking about future travel plans, or neither.</thinking>
    <status>If client is currently traveling, say CURRENTLY TRAVELING, otherwise say NOT CURRENTLY TRAVELING</status>
</current_travel_status>

<health_impediments>
    <impediment>List each separate condition that could affect exercise/nutrition, including:
    - Injuries (acute or chronic)
    - Recent medical procedures (including vaccinations)
    - Illnesses or infections
    - Physical limitations or persistent issues
    - Body pain or discomfort
    - Fatigue or energy issues</impediment>
</health_impediments>

<todos>
    <todo>One specific todo mention per tag</todo>
</todos>

Context: {text_messages}
Staff Notes: {internal_notes}
"""  # noqa
    assistant = "<current_travel_status>"

    @classmethod
    def format_prompt(cls, text_messages: list[str], internal_notes: list[str]) -> dict:
        text_messages_str = "\n".join(text_messages)
        internal_notes_str = "\n".join(internal_notes)
        return {
            "system": cls.system,
            "user": cls.user.format(text_messages=text_messages_str, internal_notes=internal_notes_str),
            "assistant": cls.assistant,
        }

    @classmethod
    def parse_response(cls, response: Any, model: str) -> tuple[str, float]:
        response, cost = text_cost_parser(response, model)
        response = cls.assistant + response
        return response, cost


class SynthesizeInjuriesPrompt:
    system = """
You are a precise medical record synthesizer. Your goal is to maintain a concise, current snapshot of active health conditions.

Key principles:
1. Maintain maximum brevity while preserving crucial medical information
2. Remove resolved conditions completely
3. Combine related conditions
4. Only include current/ongoing conditions
5. Strip out unnecessary narrative elements

Length limit: The synthesis should never exceed 200 words.
"""

    user = """
CONTEXT
Previous State ({old_date_range}): {previous_synthesis}
New Data ({new_date_range}): {data}

REQUIRED OUTPUT FORMAT
<active_conditions>
[Condition Name]: [Current Status] (Only include date for major events)
</active_conditions>

Instructions:
1. Review previous state
2. Update with new information
3. REMOVE:
   - Resolved conditions
   - Redundant information
   - Historical context unless critical
   - Unnecessary narrative details
4. COMBINE:
   - Related symptoms under primary condition
   - Multiple occurrences of same condition
5. FORMAT:
   - Use bullet points
   - One line per condition when possible
   - Only include dates for surgeries/major diagnoses

Example format:
• Left Knee ACL Tear: Post-surgery rehabilitation (Surgery: 2024-01-15)
• Chronic Lower Back Pain: Managing with PT, moderate improvement
"""

    assistant = "<current_active_conditions>"

    @classmethod
    def format_prompt(cls, data: dict, previous_synthesis: str = "", old_date_range: str = "") -> dict:
        date = list(data.keys())[0]
        info = data[date]

        injuries = info.get("health_impediments", [])
        data_str = "\n".join(injuries) if injuries else "No health impediments reported."

        return {
            "system": cls.system,
            "user": cls.user.format(
                previous_synthesis=previous_synthesis or "None",
                old_date_range=old_date_range or "None",
                new_date_range=date,
                data=data_str,
            ),
            "assistant": cls.assistant,
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class SynthesizeGoalsPrompt:
    system = """
You are a precise health goals synthesizer. Your role is to maintain a focused, current snapshot of a client's active health priorities and goals.

Key principles:
1. Maintain maximum clarity while preserving specific, measurable targets
2. Remove completed or abandoned goals completely
3. Combine related goals under primary objectives
4. Only include currently pursued goals and priorities
5. Strip out unnecessary context and commentary

Length limit: Maximum 3 priority areas, each with no more than 1-2 specific goals.
"""  # noqa

    user = """
CONTEXT
Original Profile Goals: {profile}
Previous Week's Goals ({first_date} : {old_date}): {previous_synthesis}
New Data ({new_date}):
{text_messages}
{internal_notes}
{micro_goals}

REQUIRED OUTPUT FORMAT
<current_goals>
Priority Area
• Primary Goal: [Specific target/metric]
  - Current focus: [Immediate action items]
</current_goals>

Instructions:
1. Review previous goals and new information
2. REMOVE:
   - Completed goals
   - Abandoned priorities
   - Vague or unmeasurable objectives
   - Goals not mentioned in recent weeks
3. COMBINE:
   - Related goals under main priorities
   - Overlapping objectives
4. MAINTAIN:
   - Specific metrics and targets
   - Clear timelines when mentioned
   - Active progress tracking
5. FORMAT:
   - Maximum 3 priority areas
   - 1-2 specific goals per priority
   - Include metrics when available

Example format:
Strength Training
• Primary Goal: Increase deadlift to 225lbs by September
  - Current focus: Form improvement, 3x weekly sessions

Sleep Optimization
• Primary Goal: Achieve 7+ hours sleep 6 nights/week
  - Current focus: 10:30pm bedtime routine
"""  # noqa

    assistant = "<current_goals>"

    @classmethod
    def format_prompt(
        cls,
        first_date: str,
        profile: str,
        date: str,
        text_messages: str,
        internal_notes: str,
        micro_goals: str,
        previous_synthesis: str = "",
        old_date_range: str = "",
    ) -> dict:
        return {
            "system": cls.system,
            "user": cls.user.format(
                first_date=first_date,
                profile=profile,
                text_messages=text_messages,
                internal_notes=internal_notes,
                micro_goals=micro_goals,
                new_date=date,
                previous_synthesis=previous_synthesis or "None",
                old_date=old_date_range or "None",
            ),
            "assistant": cls.assistant,
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)


class LabelInjuryClustersPrompt:
    system = "You are an expert at analyzing health data and labeling clusters."
    user = """
# Task Overview
You are an expert at analyzing health data and labeling clusters. You will analyze health conditions and output structured XML data following strict formatting rules.

# Output Format Requirements
Your response must be a single <structuredOutput> element containing:
- <label>: A precise medical or descriptive term
- <description>: Contains three sub-elements:
  - <symptoms>: Primary manifestations
  - <affectedAreas>: Anatomical locations
  - <causes>: Known triggers or causes
- <severity>: Contains two sub-elements:
  - <temporalClass>: Either "Acute" or "Chronic"
  - <impactLevel>: Integer 1-5 following the scale below

# Severity Scale
Impact Level criteria:
1: Minimal impact (minor discomfort, no functional limitation)
2: Mild impact (some discomfort, minimal functional limitation)
3: Moderate impact (notable discomfort, moderate functional limitation)
4: Severe impact (significant pain/disability, major functional limitation)
5: Critical impact (extreme pain/disability, severe functional limitation)

# Example Valid Output
<structuredOutput>
  <label>Chronic Knee Osteoarthritis</label>
  <description>
    <symptoms>Persistent pain, swelling, and limited range of motion</symptoms>
    <affectedAreas>Right knee joint, specifically medial compartment</affectedAreas>
    <causes>Progressive joint degeneration, exacerbated by weight-bearing activities</causes>
  </description>
  <severity>
    <temporalClass>Chronic</temporalClass>
    <impactLevel>4</impactLevel>
  </severity>
</structuredOutput>

# Validation Rules
1. All tags must be properly nested and closed
2. Label must use specific medical terminology
3. Description must include all three sub-elements
4. temporalClass must be exactly "Acute" or "Chronic"
5. impactLevel must be an integer between 1-5
6. Multiple related conditions should be consolidated under the primary condition
7. All text content must be properly escaped for XML

Input Text: {entries}

"""  # noqa
    assistant = "<structuredOutput>"

    @classmethod
    def format_prompt(cls, entries: list[dict]) -> dict:
        entries_str = "\n".join([e["text"].strip() for e in entries])
        return {"system": cls.system, "user": cls.user.format(entries=entries_str), "assistant": cls.assistant}

    @classmethod
    def parse_response(cls, response: Any, model: str) -> tuple[str, float]:
        response, cost = text_cost_parser(response, model)
        response = cls.assistant + response + "</structuredOutput>"
        root = ET.fromstring(response)

        result = {
            "label": root.find("label").text,
            "description": {
                "symptoms": root.find("description/symptoms").text,
                "affectedAreas": root.find("description/affectedAreas").text,
                "causes": root.find("description/causes").text,
            },
            "severity": {
                "temporalClass": root.find("severity/temporalClass").text,
                "impactLevel": int(root.find("severity/impactLevel").text),
            },
        }

        return result, cost


class LabelMicroGoalClustersPrompt:
    system = "You are an expert at analyzing behavioral patterns and labeling goal clusters."
    user = """
# Task Overview
You will analyze clusters of health-related microgoals and output structured XML data that categorizes and describes these goal patterns.

# Output Format Requirements
Your response must be a single <structuredOutput> element containing:
- <label>: A concise category name for this type of goal
- <description>: Contains four sub-elements:
  - <objective>: The primary intended outcome
  - <actionArea>: The specific health/lifestyle domain
  - <approach>: The method or strategy used
  - <complexity>: Integer 1-5 following the scale below

# Complexity Scale
Level criteria:
1: Very Simple (single, straightforward action)
2: Simple (basic habit formation, minimal planning)
3: Moderate (multiple steps or tracking required)
4: Complex (requires significant planning/tracking)
5: Very Complex (multiple interconnected behaviors/tracking)

# Example Valid Output
<structuredOutput>
  <label>Daily Water Intake Increase</label>
  <description>
    <objective>Improve hydration through consistent water consumption</objective>
    <actionArea>Nutrition and hydration habits</actionArea>
    <approach>Regular water intake tracking and scheduling</approach>
    <complexity>2</complexity>
  </description>
</structuredOutput>

# Validation Rules
1. All tags must be properly nested and closed
2. Label must be clear and behavior-focused
3. Description must include all three sub-elements
4. complexity must be an integer between 1-5
5. All text content must be properly escaped for XML

Input Text:\n{entries}
"""  # noqa
    assistant = "<structuredOutput>"

    @classmethod
    def format_prompt(cls, entries: list[dict]) -> dict:
        entries_str = "\n".join([e["text"].strip() for e in entries])
        return {"system": cls.system, "user": cls.user.format(entries=entries_str), "assistant": cls.assistant}

    @classmethod
    def parse_response(cls, response: Any, model: str) -> tuple[str, float]:
        response, cost = text_cost_parser(response, model)
        response = cls.assistant + response + "</structuredOutput>"
        root = ET.fromstring(response)
        result = {
            "label": root.find("label").text,
            "description": {
                "objective": root.find("description/objective").text,
                "actionArea": root.find("description/actionArea").text,
                "approach": root.find("description/approach").text,
                "complexity": int(root.find("description/complexity").text),
            },
        }
        return result, cost


class ClientTrendsPrompt2:
    """
    A prompt class for identifying client pain points and trends.
    """

    system_prompt = """
    You are an intelligent assistant specialized in analyzing client communications.
    Your task is to analyze the text and email conversations between the team and the client for the week.
    Extract specific information and structure your response in XML format as outlined below.
    """

    user_prompt = """
    Please analyze the following client messages and team communications for the week. {notes}
    Extract the following information:

    1. **Client Focused Topics**: Identify which of the following topics the CLIENT is focusing on: Recovery, Nutrition, Exercise, Sleep. Classify each topic as `True` or `False`. The team may bring up these topics, but pay attention to the client's focus. You can select multiple topics.

    2. **Injuries**: Determine if the client mentions any injuries. If so, specify the type of injury.

    3. **Travel Plans**: Identify if the client is planning to travel or traveling this week. If yes, provide the travel dates if available.

    4. **Intervention Points**: Detect specific points where either the client or the team brings up important topics or concerns. For each intervention point, provide a coupled recommendation to address it. For example, "Client mentioned feeling tired during workouts. Recommend increasing sleep duration." or "Team suggested adding more protein to the client's diet."

    Be as detailed as possible in your analysis.
    Structure your response using the following XML format:

    ```xml
    <ClientAnalysis>
        <Topics>
            <Recovery>True/False</Recovery>
            <Nutrition>True/False</Nutrition>
            <Exercise>True/False</Exercise>
            <Sleep>True/False</Sleep>
        </Topics>
        <Injuries>Injury Description or None</Injuries>
        <TravelPlans>
            <IsTraveling>True/False</IsTraveling>
            <TravelDates>Start Date to End Date or None</TravelDates>
        </TravelPlans>
        <InterventionPoints>
            <Intervention>
                <Description>Description of Intervention Point</Description>
                <Recommendation>Recommendation to Alleviate</Recommendation>
            </Intervention>
            <!-- Repeat <Intervention> blocks as needed -->
        </InterventionPoints>
    </ClientAnalysis>
    """  # noqa

    DataModel = None

    @classmethod
    async def format_prompt(cls, entry: Entry, read=None):
        if read is not None:
            with open(entry.ingestion.parsed_file_path) as f:
                document = f.read()
        else:
            document = await read(entry.ingestion.parsed_file_path)
        return cls.system_prompt, cls.user_prompt.format(notes=document)

    @classmethod
    def parse_response(cls, entries: list[Entry], parsed_entries: dict[str, str]) -> dict:
        for i, entry in enumerate(entries):
            try:
                response = parsed_entries.get(i).get("response")
                # Remove any leading/trailing whitespace and non-printable characters
                response = response.strip()
                response = re.sub(r"^\s*", "", response)

                # Check if the response starts with the XML declaration or root element
                if not response.startswith(("<?xml", "<ClientAnalysis")):
                    # If not, find the start of the XML content
                    xml_start = response.find("<ClientAnalysis")
                    if xml_start == -1:
                        raise ValueError("Could not find the start of XML content")
                    response = response[xml_start:]

                # Check if the response ends with the XML closing tag
                if not response.endswith("</ClientAnalysis>"):
                    # If not, find the end of the XML content
                    xml_end = response.rfind("</ClientAnalysis>")
                    if xml_end == -1:
                        raise ValueError("Could not find the end of XML content")
                    response = response[: xml_end + len("</ClientAnalysis>")]

                # remove the words xml and xml version
                root = ET.fromstring(response)

                # Parse Topics
                topics = {
                    "Recovery": root.findtext("./Topics/Recovery") == "True",
                    "Nutrition": root.findtext("./Topics/Nutrition") == "True",
                    "Exercise": root.findtext("./Topics/Exercise") == "True",
                    "Sleep": root.findtext("./Topics/Sleep") == "True",
                }

                # Parse Injuries
                injuries = root.findtext("./Injuries")
                if injuries == "None":
                    injuries = None

                # Parse Travel Plans
                is_traveling_text = root.findtext("./TravelPlans/IsTraveling")
                is_traveling = is_traveling_text == "True"
                travel_dates = root.findtext("./TravelPlans/TravelDates")
                if travel_dates == "None":
                    travel_dates = None

                # Parse Intervention Points
                intervention_points = []
                for intervention in root.findall("./InterventionPoints/Intervention"):
                    description = intervention.findtext("Description")
                    recommendation = intervention.findtext("Recommendation")
                    intervention_points.append({"description": description, "recommendation": recommendation})

                # Create DataModel instance
                data = {
                    "topics": topics,
                    "injuries": injuries,
                    "is_traveling": is_traveling,
                    "travel_dates": travel_dates,
                    "intervention_points": intervention_points,
                }
            except ET.ParseError:
                data = {"response": response}
            except ValueError:
                data = None

            entry.added_featurization = data

        return entries

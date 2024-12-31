import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional
import json
from pydantic import BaseModel, Field
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).parents[2]))

from src.prompts.parser import text_cost_parser, structured_text_cost_parser
from src.prompts.registry import PromptRegistry
from src.schemas.schemas import Entry


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    try:
        return re.sub('<[^<]+?>', '', text).strip()
    except Exception:
        return text


def calculate_age(birth_date: str) -> int:
    """Calculate age from birth date string."""
    try:
        birth = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
        today = datetime.now(timezone.utc)
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except Exception:
        return 0


def format_historical_context(context: list[dict], top_k: int) -> str:
    """Format historical context entries."""
    sorted_context = sorted(context[:top_k], key=lambda x: x["date"])
    return "\n".join(
        f"[{item['date']}] {item['title'].upper()}:\n{item['content']}"
        for item in sorted_context
    )


@PromptRegistry.register("assessment_review")
class AssessmentReviewPrompt:
    system = """You are an advanced AI exercise physiologist and health coach analyzing highly personalized exercise recommendations. Your role is to explain how the proposed schedule has been precisely engineered to the client's:

1. Current fitness level and exercise capacity
2. Health markers and clinical metrics
3. Personal goals and long-term aspirations
4. Exercise history and biomechanical preferences

**Tone:** Ensure that all recommendations and rationales are presented in a client-facing, clear, and supportive manner. The language should be accessible, encouraging, and free of technical jargon to facilitate the client's understanding and engagement.

Focus on creating a clear, sophisticated explanation that helps the client understand:
- Why this schedule is uniquely engineered for their profile
- How it will systematically progress them toward their goals
- What makes the intensity progression scientifically appropriate for them"""

    user = """Based on this client's profile, provide a comprehensive assessment review that aligns with their needs and goals.

    - Their blood markers:
     • LDL Cholesterol: {ldl_value} {ldl_unit} ({ldl_level})
     • HDL Cholesterol: {hdl_value} {hdl_unit} ({hdl_level})
     • Triglycerides: {trig_value} {trig_unit} ({trig_level})
     • HbA1c: {hba1c_value} {hba1c_unit} ({hba1c_level})
   - Their fitness metrics:
     • VO2 Peak: {vo2_value} {vo2_unit} ({vo2_level})
     • Body Fat: {body_fat}% ({body_fat_level})
     • Grip Strength: {grip_strength} {grip_unit} ({grip_level})
     • Sit-to-Stand: {sit_to_stand} {sit_to_stand_unit} ({sit_to_stand_level})
     • Movement Quality: {movement_quality} ({movement_quality_level})
   - Their health span scores:
     • Aerobic Fitness: {aerobic_fitness}/100
     • Muscular Fitness: {muscular_fitness}/100
     • Joint Fitness: {joint_fitness}/100
     • Balance: {balance}/100
     • Body Composition: {body_composition}/100
     • Cognitive Health: {cognitive_health}/100
   - Their reported joint status: "{joint_status}"
   - Their exercise adaptation level: "{exercise_adaptation}"
   - Their primary goal of "{primary_goal}"
   - Their stated long-term aspirations: "{long_term_aspirations}"
   {activity_rationale}
   - Their current capacity to exercise score of {cte}/100
   - Their reported availability and preferences
   {weekly_time_rationale}

    WEEKLY SCHEDULE:
    {weekly_schedule}

    Structure your response to populate the following sections of the DataModel JSON:

- **Overview:** Summarize the client's overall profile and key considerations. Keep this only 2 to 3 sentences.
- **Blood Work Rationale:** Explain how the client's blood markers influence the exercise recommendations. Include specific numerical values and their implications.
- **Fitness Assessment Rationale:** Detail how the client's fitness metrics inform the schedule design. Include specific values and their relevance.
- **Activity Selection Logic:** Describe the reasoning behind the chosen activities and their alignment with the client's goals.
- **Light Week Rationale:** Justify the structure and intensity of the light week.
- **Moderate Week Rationale:** Justify the structure and intensity of the moderate week.
- **Vigorous Week Rationale:** Justify the structure and intensity of the vigorous week.
- **Safety Considerations:** Outline any safety measures and ramp-up logic incorporated into the schedule.

**Additional Instructions:**
- Reference specific client data throughout each section, demonstrating personalization.
- Avoid repeating the weekly schedule in your response.
- Do not respond in Markdown; return a structured JSON as defined by the DataModel.
- Ensure the tone is client-facing, clear, and supportive."""

    class DataModel(BaseModel):
        overview: str = Field(..., title="Client Overview")
        blood_work_rationale: str = Field(..., title="Blood Work Rationale")
        fitness_assessment_rationale: str = Field(..., title="Fitness Assessment Rationale")
        activity_selection_logic: str = Field(..., title="Activity Selection Logic")
        light_week_rationale: str = Field(..., title="Light Week Rationale")
        moderate_week_rationale: str = Field(..., title="Moderate Week Rationale")
        vigorous_week_rationale: str = Field(..., title="Vigorous Week Rationale")
        safety_considerations: str = Field(..., title="Safety Considerations and Ramp-up Logic")

    @classmethod
    def format_weekly_schedule(cls, schedule: dict) -> str:
        if not schedule:
            return "No schedule provided"
        formatted = []
        for intensity in ['light', 'moderate', 'vigorous']:
            week = schedule.get(intensity, {})
            activities = []
            for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                day_activities = week.get(day, [])
                if day_activities:
                    activities.append(f"{day.capitalize()}:")
                    for activity in day_activities:
                        activities.append(f"  • {activity.get('name', 'no data available')}: {activity.get('time', 'no data available')} minutes (RPE: {activity.get('rpe', 'no data available')})")
            if activities:
                formatted.append(f"{intensity.upper()} WEEK:\n" + "\n".join(activities))
        return "\n\n".join(formatted)

    @classmethod
    def format_prompt(cls, client_data: dict, weekly_schedule: dict, activities_rationale: str, weekly_allocation_rationale: str) -> dict:
        """Format client data and exercise schedule into a structured prompt."""
        formatted_schedule = cls.format_weekly_schedule(weekly_schedule)

        # Get health span metrics
        health_span = client_data.get('assessments', {}).get('healthSpan', {})

        # Get blood metrics
        blood_metrics = client_data.get('bloodMetrics', {})
        core_lipids = blood_metrics.get('coreLipids', {})
        blood_sugar = blood_metrics.get('bloodSugar', {})

        return {
            "system": cls.system,
            "user": cls.user.format(
                weekly_schedule=formatted_schedule,
                activity_rationale=activities_rationale,
                weekly_time_rationale=weekly_allocation_rationale,

                # Blood markers
                ldl_value=core_lipids.get('LDL - Cholesterol', {}).get('value', 'no data available'),
                ldl_unit=core_lipids.get('LDL - Cholesterol', {}).get('unit', 'no data available'),
                ldl_level=core_lipids.get('LDL - Cholesterol', {}).get('level', 'no data available'),

                hdl_value=core_lipids.get('HDL - Cholesterol', {}).get('value', 'no data available'),
                hdl_unit=core_lipids.get('HDL - Cholesterol', {}).get('unit', 'no data available'),
                hdl_level=core_lipids.get('HDL - Cholesterol', {}).get('level', 'no data available'),

                trig_value=core_lipids.get('Triglycerides', {}).get('value', 'no data available'),
                trig_unit=core_lipids.get('Triglycerides', {}).get('unit', 'no data available'),
                trig_level=core_lipids.get('Triglycerides', {}).get('level', 'no data available'),

                hba1c_value=blood_sugar.get('Hemoglobin A1c', {}).get('value', 'no data available'),
                hba1c_unit=blood_sugar.get('Hemoglobin A1c', {}).get('unit', 'no data available'),
                hba1c_level=blood_sugar.get('Hemoglobin A1c', {}).get('level', 'no data available'),

                # Basic metrics
                cte=client_data.get("assessments", {}).get("lifestyle", {}).get("cte", 'no data available'),
                primary_goal=client_data.get("goals", {}).get("primary_goal", 'no data available'),
                long_term_aspirations=strip_html_tags(client_data.get("person", {}).get("long_term_aspirations", 'no data available')),

                # Survey responses
                joint_status=client_data.get("survey", {}).get("joint_status", {}).get("value", 'no data available'),
                exercise_adaptation=client_data.get("survey", {}).get("coping_with_novel_physical_stress", {}).get("value", 'no data available'),

                # Fitness metrics
                vo2_value=client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('value', 'no data available'),
                vo2_unit=client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('unit', 'no data available'),
                vo2_level=client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('level', 'no data available'),

                body_fat=client_data.get('bodyCompositionMetrics', {}).get('bodyFat', {}).get('value', 'no data available'),
                body_fat_level=client_data.get('bodyCompositionMetrics', {}).get('bodyFat', {}).get('level', 'no data available'),

                grip_strength=client_data.get('muscleStrengthMetrics', {}).get('gripStrength', 'no data available'),
                grip_unit=client_data.get('muscleStrengthMetrics', {}).get('gripStrengthUnit', 'no data available'),
                grip_level=client_data.get('muscleStrengthMetrics', {}).get('gripStrengthLevel', 'no data available'),

                sit_to_stand=client_data.get('muscleStrengthMetrics', {}).get('sitToStand', 'no data available'),
                sit_to_stand_unit=client_data.get('muscleStrengthMetrics', {}).get('sitToStandUnit', 'no data available'),
                sit_to_stand_level=client_data.get('muscleStrengthMetrics', {}).get('sitToStandLevel', 'no data available'),

                movement_quality=client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexion', 'no data available'),
                movement_quality_level=client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexionLevel', 'no data available'),

                # Health span scores
                aerobic_fitness=health_span.get('aerobic_fitness', 'no data available'),
                muscular_fitness=health_span.get('muscular_fitness', 'no data available'),
                joint_fitness=health_span.get('joint_fitness', 'no data available'),
                balance=health_span.get('balance', 'no data available'),
                body_composition=health_span.get('body_composition', 'no data available'),
                cognitive_health=health_span.get('cognitive_health', 'no data available')
            )
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = structured_text_cost_parser(response, model)
        return response, cost


@PromptRegistry.register("weekly_time_allocation")
class WeeklyTimeAllocationPrompt:
    system = """You are an expert exercise physiologist specializing in time management and exercise programming. Your role is to analyze client data and recommend appropriate weekly time allocations for exercise across three different intensity weeks:

1. Light Week: Minimal training load for recovery, travel, or busy periods
2. Moderate Week: Standard training load that can be sustained long-term
3. Vigorous Week: Higher training load for focused progression periods

Your recommendations should:
- Consider the client's current exercise routine and satisfaction
- Account for their goals and aspirations
- Factor in their exercise capacity and fitness metrics
- Be realistic given their time availability and motivation
- Include clear rationale linking the recommendations to their data

Provide your response in the following format:
{
    "rationale": "Clear explanation of your reasoning...",
    "total_time_light_weekly": X,    // Total minutes for a light week
    "total_time_moderate_weekly": Y,  // Total minutes for a moderate week
    "total_time_vigorous_weekly": Z   // Total minutes for a vigorous week
}"""

    user = """Based on the following client data, recommend appropriate weekly time allocations for exercise across light, moderate, and vigorous training weeks.

{formatted_data}

Consider:
1. How their current schedule and satisfaction levels inform baseline volume
2. How their goals and aspirations suggest needed progression
3. What their exercise capacity and fitness metrics indicate about training tolerance
4. What their time availability and motivation suggest about realistic maximums
5. How many hours per day and days per week this becomes

Provide specific minute totals for each week type and explain your reasoning.
Please cite specific data from the client profile in your rationale, using the profile, survey questions, and actual raw assessment data as needed.
You should refer to specific answers, blood tests, fitness assessments, and other relevant data points in your rationale.
"""

    class DataModel(BaseModel):
        rationale: str
        total_time_light_weekly: int
        total_time_moderate_weekly: int
        total_time_vigorous_weekly: int

    @classmethod
    def format_prompt(cls, client_data: dict) -> dict:

        current_routine = f"""
Weekly Exercise Schedule:
{client_data.get('survey', {}).get('typical_week_of_exercise', 'no data available')}

Exercise Satisfaction:
• Current satisfaction with weekly exercise: {client_data.get('survey', {}).get('satisfied_with_weekly_exercise', {}).get('value', 'no data available')}/5
• Feeling about optimal exercise habits: {client_data.get('survey', {}).get('optimal_exercise_activity_habits', {}).get('value', 'no data available')}/5"""

        # Format personal profile
        profile = f"""The client is a {calculate_age(client_data.get("person", {}).get("date_of_birth", 'no data available'))}-year-old {client_data.get("person", {}).get("gender", 'no data available')}, 
{client_data.get("person", {}).get("height_inch", 'no data available')} inches tall and weighing {client_data.get("person", {}).get("weight_lbs", 'no data available')} pounds.

Exercise Background:
• Training History: {client_data.get("survey", {}).get("training_history", {}).get('value', 'no data available')}
• Joint Status: {client_data.get("survey", {}).get("joint_status", {}).get('value', 'no data available')}
• Health Status: {client_data.get("survey", {}).get("health", {}).get('value', 'no data available')}
• Exercise Adaptation: {client_data.get("survey", {}).get("coping_with_novel_physical_stress", {}).get('value', 'no data available')}

Motivation and Availability:
• Motivation Level: {client_data.get("survey", {}).get("motivation", {}).get('value', 'no data available')}
• Time Availability: {client_data.get("survey", {}).get("time_availability", {}).get('value', 'no data available')}

Goals (in priority order):
1. {client_data.get("goals", {}).get("primary_goal", 'no data available')}
2. {client_data.get("goals", {}).get("secondary_goal", 'no data available')}
3. {client_data.get("goals", {}).get("tertiary_goal", 'no data available')}"""

        # Format detailed fitness assessment
        fitness = f"""
    Aerobic Capacity:
    • VO2 Peak: {client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('value', 'no data available')} {client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('unit', 'no data available')} ({client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('level', 'no data available')})
    Context: Key indicator of cardiorespiratory fitness and mortality risk. Higher values indicate better aerobic capacity.

    • Max Heart Rate: {client_data.get("aerobicMetrics", {}).get("maxHeartRate", 'no data available')} bpm
    Context: Used to determine appropriate training zones for cardiovascular exercise.

    Body Composition:
    • Body Fat: {client_data.get('bodyCompositionMetrics', {}).get('percent_fat_value', 'no data available')}% ({client_data.get('bodyCompositionMetrics', {}).get('percent_fat_level', 'no data available')})
    Context: Indicates overall metabolic health and fitness. Optimal ranges vary by age and gender.

    Strength and Functional Assessments:
    • Grip Strength: {client_data.get('muscleStrengthMetrics', {}).get('gripStrength', 'no data available')} {client_data.get('muscleStrengthMetrics', {}).get('gripStrengthUnit', 'no data available')} ({client_data.get('muscleStrengthMetrics', {}).get('gripStrengthLevel', 'no data available')})
    Context: Powerful predictor of overall strength and longevity. Strong correlation with all-cause mortality.

    • Sit-to-Stand Test: {client_data.get('muscleStrengthMetrics', {}).get('sitToStand', 'no data available')} {client_data.get('muscleStrengthMetrics', {}).get('sitToStandUnit', 'no data available')} ({client_data.get('muscleStrengthMetrics', {}).get('sitToStandLevel', 'no data available')})
    Context: Measures lower body strength and functional mobility. Important for daily activities and independence.

    • Movement Quality: {client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexion', 'no data available')} ({client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexionLevel', 'no data available')})
    Context: Assesses overall movement quality and joint mobility. Critical for injury prevention and functional movement."""

        # Format lifestyle metrics
        exercise_capacity = f"""
    Exercise Capacity:
    • Current Capacity to Exercise (CTE): {client_data.get("assessments", {}).get("lifestyle", {}).get("cte", 'no data available')}/100
    Context: Indicates ability to handle exercise load and intensity. Higher scores suggest better tolerance for varied training."""

        # Format the complete prompt
        formatted_data = f"""
        CLIENT PROFILE:
        {profile}

        CURRENT EXERCISE ROUTINE:
        {current_routine}

        DETAILED FITNESS ASSESSMENT:
        {fitness}

        CAPACITY TO EXERCISE:
        {exercise_capacity}
        """

        return {
            "system": cls.system,
            "user": cls.user.format(formatted_data=formatted_data)
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = structured_text_cost_parser(response, model)
        return response, cost


@PromptRegistry.register("weekly_time_allocation_critic")
class WeeklyTimeAllocationCriticPrompt:
    system = """You are an expert exercise physiologist specializing in time management and exercise programming. Your role is to analyze three different weekly time allocation recommendations and synthesize them into a final recommendation that captures the best insights while addressing any inconsistencies.

Your task is to:
1. Review three different time allocation proposals
2. Identify common patterns and recommendations
3. Flag any concerning discrepancies or potential errors
4. Synthesize the recommendations into a final proposal that:
   - Maintains consistency with the client's profile
   - Incorporates the strongest reasoning from each proposal
   - Resolves any conflicts between recommendations
   - Ensures realistic and achievable targets

Provide your response in the following format:
{
    "rationale": "Analysis of the proposals and explanation of final synthesis...",
    "display_text": "string",         // Detailed explanation of the final recommendation without references to the proposals
    "total_time_light_weekly": X,    // Synthesized total minutes for light week
    "total_time_moderate_weekly": Y,  // Synthesized total minutes for moderate week
    "total_time_vigorous_weekly": Z   // Synthesized total minutes for vigorous week
}"""

    user = """Review these three weekly time allocation proposals for the client:

PROPOSAL 1:
{proposal_1}

PROPOSAL 2:
{proposal_2}

PROPOSAL 3:
{proposal_3}

CLIENT CONTEXT:
{formatted_data}

Analyze the proposals for:
1. Common patterns in recommended time allocations
2. Significant discrepancies between recommendations
3. How well each aligns with the client's profile
4. Potential errors or unrealistic suggestions

Synthesize these into a final recommendation that:
- Captures the most well-reasoned aspects of each proposal
- Resolves any conflicts or inconsistencies
- Ensures alignment with client capacity and goals
- Maintains realistic and achievable targets

Please cite specific data from the client profile in your display_text, using the profile, survey questions, and actual raw assessment data as needed.
You should refer to specific answers, blood tests, fitness assessments, and other relevant data points in the display_text.
Provide your analysis and final recommendation in the specified JSON format."""

    class DataModel(BaseModel):
        rationale: str
        display_text: str
        total_time_light_weekly: int
        total_time_moderate_weekly: int
        total_time_vigorous_weekly: int

    @classmethod
    def format_prompt(cls, client_data: dict, proposals: list[dict]) -> dict:
        # Format personal profile (simplified for critic)
        profile = f"""The client is a {calculate_age(client_data.get("person", {}).get("date_of_birth", 'no data available'))}-year-old {client_data.get("person", {}).get("gender", 'no data available')}.

Exercise Background:
• Training History: {client_data.get("survey", {}).get("training_history", {}).get("value", 'no data available')}
• Joint Status: {client_data.get("survey", {}).get("joint_status", {}).get("value", 'no data available')}
• Exercise Adaptation: {client_data.get("survey", {}).get("coping_with_novel_physical_stress", {}).get("value", 'no data available')}

Motivation and Availability:
• Motivation Level: {client_data.get("survey", {}).get("motivation", {}).get("value", 'no data available')}
• Time Availability: {client_data.get("survey", {}).get("time_availability", {}).get("value", 'no data available')}

Current Exercise Satisfaction:
• Satisfaction with weekly exercise: {client_data.get('survey', {}).get('satisfied_with_weekly_exercise', {}).get('value', 'no data available')}/5
• Feeling about optimal habits: {client_data.get('survey', {}).get('optimal_exercise_activity_habits', {}).get('value', 'no data available')}/5

Exercise Capacity:
• Current Capacity to Exercise (CTE): {client_data.get("assessments", {}).get("lifestyle", {}).get("cte", 'no data available')}/100"""

        # Format the proposals
        proposal_texts = []
        for i, prop in enumerate(proposals[:3], 1):
            proposal_texts.append(f"""
Rationale: {prop.get('rationale', 'no data available')}
Light Week: {prop.get('total_time_light_weekly', 'no data available')} minutes
Moderate Week: {prop.get('total_time_moderate_weekly', 'no data available')} minutes
Vigorous Week: {prop.get('total_time_vigorous_weekly', 'no data available')} minutes""")

        return {
            "system": cls.system,
            "user": cls.user.format(
                proposal_1=proposal_texts[0],
                proposal_2=proposal_texts[1],
                proposal_3=proposal_texts[2],
                formatted_data=profile
            )
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = structured_text_cost_parser(response, model)
        return response, cost


@PromptRegistry.register("activity_list")
class ActivityListPrompt:
    activity_categories = [
        'Elliptical', 'Barre', 'Cross Training', 'Cycling', 'Meditation', 
        'Pickleball', 'Squash', 'Tennis', 'Golf', 'Rowing', 'Pilates', 
        'Rugby', 'Running', 'Strength', 'Swimming', 'Walking', 'Yoga'
    ]

    system = """You are an expert exercise physiologist specializing in activity selection and program design. Your role is to analyze client data and recommend appropriate activities based on their profile, preferences, and needs.

Your task is to categorize activities into four lists:
1. Activities the client likely enjoys (based on current routine and stated preferences)
2. Activities the client likely dislikes or should avoid (based on limitations, injuries, or stated preferences)
3. Activities the client should increase (based on goals, health metrics, and areas needing improvement)
4. Activities the client should potentially decrease (only if their schedule is overly focused on certain modalities)

Consider:
- Current exercise routine and satisfaction levels
- Health metrics and areas needing improvement
- Personal goals and aspirations
- Physical limitations and joint status
- Exercise history and adaptation capacity

Provide your response in the following JSON format:
{
    "likes_activities": ["activity1", "activity2", ..., "activityN"],     // Activities client currently does or would likely enjoy
    "dislikes_activities": ["activity3", "activity4", ..., "activityM"],  // Activities to avoid based on limitations or preferences
    "increase_activities": ["activity5", "activity6", ..., "activityK"],  // Activities that would benefit their goals/health
    "decrease_activities": ["activity7", ..., "activityL"]                // Activities to reduce if overemphasized
}
"""

    user = """Based on the following client data, recommend appropriate activities that align with their profile, goals, and health status.

{formatted_data}

Only use activities from the following categories:
{activity_categories}

Consider:
1. Which activities are currently in their routine or align with their preferences. Be thorough in combing through the data to identify these.
2. Which activities might be contraindicated based on their joint status or limitations. Cross Training includes HITT, circuit training, and other mixed modalities.
3. Which activities would help improve their lower-scoring health domains
4. Which activities best support their stated goals and aspirations
5. Whether any current activities are overemphasized and should be reduced

Be careful adding sports, since some may not play or like them. Only include sports if they are explicitly mentioned in the data. The general categories are fine to use.

Provide your recommendations in the specified JSON format using only the allowed activity categories.
Please cite specific data from the client profile in your rationale, using the profile, survey questions, and actual raw assessment data as needed.
You should refer to specific answers, blood tests, fitness assessments, and other relevant data points in the rationale.
"""

    class DataModel(BaseModel):
        likes_activities: list[str]
        dislikes_activities: list[str]
        increase_activities: list[str]
        decrease_activities: list[str]

    @classmethod
    def format_prompt(cls, client_data: dict) -> dict:

        current_routine = f"""
Weekly Exercise Schedule:
{client_data.get('survey', {}).get('typical_week_of_exercise', 'no data available')}

Exercise Satisfaction:
• Current satisfaction with weekly exercise: {client_data.get('survey', {}).get('satisfied_with_weekly_exercise', {}).get('value', 'no data available')}/5
• Feeling about optimal exercise habits: {client_data.get('survey', {}).get('optimal_exercise_activity_habits', {}).get('value', 'no data available')}/5"""

        # Format personal profile
        profile = f"""The client is a {calculate_age(client_data.get("person", {}).get("date_of_birth", 'no data available'))}-year-old {client_data.get("person", {}).get("gender", 'no data available')}, 
{client_data.get("person", {}).get("height_inch", 'no data available')} inches tall and weighing {client_data.get("person", {}).get("weight_lbs", 'no data available')} pounds.

Exercise Background:
• Training History: {client_data.get("survey", {}).get("training_history", {}).get("value", 'no data available')}
• Joint Status: {client_data.get("survey", {}).get("joint_status", {}).get("value", 'no data available')}
• Health Status: {client_data.get("survey", {}).get("health", {}).get("value", 'no data available')}
• Exercise Adaptation: {client_data.get("survey", {}).get("coping_with_novel_physical_stress", {}).get("value", 'no data available')}

Motivation and Availability:
• Motivation Level: {client_data.get("survey", {}).get("motivation", {}).get("value", 'no data available')}
• Time Availability: {client_data.get("survey", {}).get("time_availability", {}).get("value", 'no data available')}

Goals (in priority order):
1. {client_data.get("goals", {}).get("primary_goal", 'no data available')}
2. {client_data.get("goals", {}).get("secondary_goal", 'no data available')}
3. {client_data.get("goals", {}).get("tertiary_goal", 'no data available')}

Long-term aspirations: {strip_html_tags(client_data.get("person", {}).get("long_term_aspirations", 'no data available'))}
Areas of focus: {strip_html_tags(client_data.get("person", {}).get("areas_of_focus", 'no data available'))}"""

        # Format health span metrics
        health_span = f"""
    Overall Health Metrics:
    • Aerobic Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("aerobic_fitness", 'no data available')}/100
    Context: Measures cardiovascular endurance and oxygen utilization efficiency.

    • Body Composition: {client_data.get("assessments", {}).get("healthSpan", {}).get("body_composition", 'no data available')}/100
    Context: Reflects distribution of muscle, fat, and other tissues.

    • Muscular Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("muscular_fitness", 'no data available')}/100
    Context: Indicates strength, power, and muscular endurance capabilities.

    • Joint Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("joint_fitness", 'no data available')}/100
    Context: Assesses joint health, mobility, and movement quality.

    • Balance: {client_data.get("assessments", {}).get("healthSpan", {}).get("balance", 'no data available')}/100
    Context: Evaluates neuromuscular control and stability.

    • Cognitive Health: {client_data.get("assessments", {}).get("healthSpan", {}).get("cognitive_health", 'no data available')}/100
    Context: Measures mental acuity and cognitive function."""

        # Format detailed fitness assessment
        fitness = f"""
    Aerobic Capacity:
    • VO2 Peak: {client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('value', 'no data available')} {client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('unit', 'no data available')} ({client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('level', 'no data available')})
    Context: Key indicator of cardiorespiratory fitness and mortality risk. Higher values indicate better aerobic capacity.

    • Max Heart Rate: {client_data.get("aerobicMetrics", {}).get("maxHeartRate", 'no data available')} bpm
    Context: Used to determine appropriate training zones for cardiovascular exercise.

    Body Composition:
    • Body Fat: {client_data.get('bodyCompositionMetrics', {}).get('percent_fat_value', 'no data available')}% ({client_data.get('bodyCompositionMetrics', {}).get('percent_fat_level', 'no data available')})
    Context: Indicates overall metabolic health and fitness. Optimal ranges vary by age and gender.

    Strength and Functional Assessments:
    • Grip Strength: {client_data.get('muscleStrengthMetrics', {}).get('gripStrength', 'no data available')} {client_data.get('muscleStrengthMetrics', {}).get('gripStrengthUnit', 'no data available')} ({client_data.get('muscleStrengthMetrics', {}).get('gripStrengthLevel', 'no data available')})
    Context: Powerful predictor of overall strength and longevity. Strong correlation with all-cause mortality.

    • Sit-to-Stand Test: {client_data.get('muscleStrengthMetrics', {}).get('sitToStand', 'no data available')} {client_data.get('muscleStrengthMetrics', {}).get('sitToStandUnit', 'no data available')} ({client_data.get('muscleStrengthMetrics', {}).get('sitToStandLevel', 'no data available')})
    Context: Measures lower body strength and functional mobility. Important for daily activities and independence.

    • Movement Quality: {client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexion', 'no data available')} ({client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexionLevel', 'no data available')})
    Context: Assesses overall movement quality and joint mobility. Critical for injury prevention and functional movement."""

        # Format the complete prompt
        formatted_data = f"""
        CLIENT PROFILE:
        {profile}

        CURRENT EXERCISE ROUTINE:
        {current_routine}

        DETAILED FITNESS ASSESSMENT:
        {fitness}

        HEALTH SPAN ASSESSMENT:
        {health_span}
        """

        return {
            "system": cls.system,
            "user": cls.user.format(formatted_data=formatted_data, activity_categories=cls.activity_categories)
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = structured_text_cost_parser(response, model)
        return response, cost


@PromptRegistry.register("activity_list_critic")
class ActivityListCriticPrompt:
    activity_categories = [
        'Elliptical', 'Barre', 'Cross Training', 'Cycling', 'Meditation',
        'Pickleball', 'Squash', 'Tennis', 'Golf', 'Rowing', 'Pilates',
        'Rugby', 'Running', 'Strength', 'Swimming', 'Walking', 'Yoga'
    ]

    system = """You are an expert exercise physiologist specializing in activity selection and program design. Your role is to analyze three different activity recommendations and synthesize them into a final recommendation that captures the best insights while addressing any inconsistencies.

Your task is to:
1. Review three different activity proposals
2. Identify common patterns and recommendations
3. Flag any concerning discrepancies or potential errors
4. Synthesize the recommendations into a final proposal that:
   - Maintains consistency with the client's profile
   - Incorporates the strongest reasoning from each proposal
   - Resolves any conflicts between recommendations
   - Ensures activities are appropriate and well-balanced

Provide your response in the following JSON format:
{
    "rationale": "Analysis of the proposals and explanation of final synthesis...",
    "display_text": "string",         // Detailed explanation of the final recommendation without references to the proposals
    "likes_activities": ["activity1", "activity2"],     // Synthesized activities client likely enjoys
    "dislikes_activities": ["activity3", "activity4"],  // Synthesized activities to avoid
    "increase_activities": ["activity5", "activity6"],  // Synthesized activities to increase
    "decrease_activities": ["activity7"]                // Synthesized activities to decrease
}
"""

    user = """Review these three activity list proposals for the client:

PROPOSAL 1:
{proposal_1}

PROPOSAL 2:
{proposal_2}

PROPOSAL 3:
{proposal_3}

CLIENT CONTEXT:
{formatted_data}

Only use activities from the following categories:
{activity_categories}

Analyze the proposals for:
1. Common activity recommendations across all proposals
2. Significant differences in activity categorization
3. How well each aligns with the client's profile
4. Potential contradictions or inappropriate suggestions

Synthesize these into a final recommendation that:
- Captures the most well-reasoned activity selections
- Resolves any conflicts in categorization
- Ensures alignment with client limitations and goals
- Maintains a balanced and appropriate activity mix

Please cite specific data from the client profile in your display_text, using the profile, survey questions, and actual raw assessment data as needed.
You should refer to specific answers, blood tests, fitness assessments, and other relevant data points in the display_text.
Provide your analysis and final recommendation in the specified JSON format."""

    class DataModel(BaseModel):
        rationale: str
        display_text: str
        likes_activities: list[str]
        dislikes_activities: list[str]
        increase_activities: list[str]
        decrease_activities: list[str]

    @classmethod
    def format_prompt(cls, client_data: dict, proposals: list[dict]) -> dict:
        # Format personal profile (simplified for critic)
        profile = f"""The client is a {calculate_age(client_data.get("person", {}).get("date_of_birth", 'no data available'))}-year-old {client_data.get("person", {}).get("gender", 'no data available')}.

Exercise Background:
• Training History: {client_data.get("survey", {}).get("training_history", {}).get("value", 'no data available')}
• Joint Status: {client_data.get("survey", {}).get("joint_status", {}).get("value", 'no data available')}
• Health Status: {client_data.get("survey", {}).get("health", {}).get("value", 'no data available')}
• Exercise Adaptation: {client_data.get("survey", {}).get("coping_with_novel_physical_stress", {}).get("value", 'no data available')}

Current Exercise Satisfaction:
• Satisfaction with weekly exercise: {client_data.get('survey', {}).get('satisfied_with_weekly_exercise', {}).get('value', 'no data available')}/5
• Feeling about optimal habits: {client_data.get('survey', {}).get('optimal_exercise_activity_habits', {}).get('value', 'no data available')}/5

Key Health Metrics:
• Aerobic Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("aerobic_fitness", 'no data available')}/100
• Muscular Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("muscular_fitness", 'no data available')}/100
• Joint Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("joint_fitness", 'no data available')}/100
• Balance: {client_data.get("assessments", {}).get("healthSpan", {}).get("balance", 'no data available')}/100

Goals (in priority order):
1. {client_data.get("goals", {}).get("primary_goal", 'no data available')}
2. {client_data.get("goals", {}).get("secondary_goal", 'no data available')}
3. {client_data.get("goals", {}).get("tertiary_goal", 'no data available')}"""

        # Format the proposals
        proposal_texts = []
        for i, prop in enumerate(proposals[:3], 1):
            proposal_texts.append(f"""
Likes: {prop.get('likes_activities', 'no data available')}
Dislikes: {prop.get('dislikes_activities', 'no data available')}
Increase: {prop.get('increase_activities', 'no data available')}
Decrease: {prop.get('decrease_activities', 'no data available')}""")

        return {
            "system": cls.system,
            "user": cls.user.format(
                proposal_1=proposal_texts[0],
                proposal_2=proposal_texts[1],
                proposal_3=proposal_texts[2],
                activity_categories=cls.activity_categories,
                formatted_data=profile
            )
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = structured_text_cost_parser(response, model)
        return response, cost


@PromptRegistry.register("healthy_signals")
class HealthySignalsPrompt:
    system = """You are an expert exercise physiologist specializing in health optimization. Your role is to analyze client data and determine target metrics for different health domains based on ACSM guidelines and RPE-based scoring.

Your task is to assign target scores for each health domain where:
- Score of 100 represents meeting ACSM guidelines and basic health targets
- Score of 200 indicates progression beyond baseline recommendations
- Score of 300 represents excellence and peak performance
- Scoring should be scaled based on the client's Capacity to Exercise (CTE):
  * CTE < 50: Target scores typically between 50-150
  * CTE 50-75: Target scores typically between 100-200
  * CTE > 75: Target scores typically between 150-300
- Scores are calculated using RPE (Rating of Perceived Exertion) derived from heart rate data:
  * Average heart rate
  * Peak heart rate
  * Maximum heart rate

Provide your response in the following JSON format:
{
    "rationale": "string",            // Thorough explanation of your reasoning
    "aerobic": float,                 // Target score (50-300)
    "bone": float,                    // Target score (50-300)
    "body_composition": float,        // Target score (50-300)
    "muscle_fitness": float,          // Target score (50-300)
    "movement_quality": float,        // Target score (50-300)
    "balance": float,                 // Target score (50-300)
    "cognitive_health": float         // Target score (50-300)
}"""

    user = """Based on the following client data, determine appropriate target scores for each health domain.

{formatted_data}

Consider:
1. Current scores relative to ACSM guidelines
2. Client's goals and potential for progression
3. Realistic achievement potential based on capacity to exercise
4. Age-appropriate targets and risk factors. Some individuals with a low capacity to exercise may not be able to even reach 100 at the start.
5. Interdependencies between domains
6. Where the client is deficient from the HEALTH SPAN ASSESSMENT. We should focus on these areas first.

For each domain:
- Set targets between 100-300
- Consider 100 as meeting basic health guidelines
- Use 200 for domains where progression beyond baseline is needed
- Reserve 300 for domains where excellence is both achievable and aligned with goals
- Account for current scores when setting realistic progression targets

Please cite specific data from the client profile in your rationale, using the profile, survey questions, and actual raw assessment data as needed.
You should refer to specific answers, blood tests, fitness assessments, and other relevant data points in the rationale.
Provide your recommendations in the specified JSON format."""

    class DataModel(BaseModel):
        rationale: str
        aerobic: float
        bone: float
        body_composition: float
        muscle_fitness: float
        movement_quality: float
        balance: float
        cognitive_health: float

    @classmethod
    def format_prompt(cls, client_data: dict) -> dict:
        # Format personal profile
        profile = f"""The client is a {calculate_age(client_data.get("person", {}).get("date_of_birth", 'no data available'))}-year-old {client_data.get("person", {}).get("gender", 'no data available')}, 
{client_data.get("person", {}).get("height_inch", 'no data available')} inches tall and weighing {client_data.get("person", {}).get("weight_lbs", 'no data available')} pounds.

Goals (in priority order):
1. {client_data.get("goals", {}).get("primary_goal", 'no data available')}
2. {client_data.get("goals", {}).get("secondary_goal", 'no data available')}
3. {client_data.get("goals", {}).get("tertiary_goal", 'no data available')}

Long-term aspirations: {strip_html_tags(client_data.get("person", {}).get("long_term_aspirations", 'no data available'))}
Areas of focus: {strip_html_tags(client_data.get("person", {}).get("areas_of_focus", 'no data available'))}

Exercise Capacity:
• Current Capacity to Exercise (CTE): {client_data.get("assessments", {}).get("lifestyle", {}).get("cte", 'no data available')}/100
Context: Indicates ability to handle exercise load and intensity. Higher scores suggest better tolerance for varied training.
"""

        # Format health span metrics
        health_span = f"""
    Overall Health Metrics:
    • Aerobic Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("aerobic_fitness", 'no data available')}/100
    Context: Measures cardiovascular endurance and oxygen utilization efficiency.

    • Body Composition: {client_data.get("assessments", {}).get("healthSpan", {}).get("body_composition", 'no data available')}/100
    Context: Reflects distribution of muscle, fat, and other tissues.

    • Muscular Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("muscular_fitness", 'no data available')}/100
    Context: Indicates strength, power, and muscular endurance capabilities.

    • Joint Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("joint_fitness", 'no data available')}/100
    Context: Assesses joint health, mobility, and movement quality.

    • Balance: {client_data.get("assessments", {}).get("healthSpan", {}).get("balance", 'no data available')}/100
    Context: Evaluates neuromuscular control and stability.

    • Cognitive Health: {client_data.get("assessments", {}).get("healthSpan", {}).get("cognitive_health", 'no data available')}/100
    Context: Measures mental acuity and cognitive function."""

        # Format detailed fitness assessment
        fitness = f"""
    Aerobic Capacity:
    • VO2 Peak: {client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('value', 'no data available')} {client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('unit', 'no data available')} ({client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('level', 'no data available')})
    Context: Key indicator of cardiorespiratory fitness and mortality risk. Higher values indicate better aerobic capacity.

    • Max Heart Rate: {client_data.get("aerobicMetrics", {}).get("maxHeartRate", 'no data available')} bpm
    Context: Used to determine appropriate training zones for cardiovascular exercise.

    Body Composition:
    • Body Fat: {client_data.get('bodyCompositionMetrics', {}).get('percent_fat_value', 'no data available')}% ({client_data.get('bodyCompositionMetrics', {}).get('percent_fat_level', 'no data available')})
    Context: Indicates overall metabolic health and fitness. Optimal ranges vary by age and gender.

    Strength and Functional Assessments:
    • Grip Strength: {client_data.get('muscleStrengthMetrics', {}).get('gripStrength', 'no data available')} {client_data.get('muscleStrengthMetrics', {}).get('gripStrengthUnit', 'no data available')} ({client_data.get('muscleStrengthMetrics', {}).get('gripStrengthLevel', 'no data available')})
    Context: Powerful predictor of overall strength and longevity. Strong correlation with all-cause mortality.

    • Sit-to-Stand Test: {client_data.get('muscleStrengthMetrics', {}).get('sitToStand', 'no data available')} {client_data.get('muscleStrengthMetrics', {}).get('sitToStandUnit', 'no data available')} ({client_data.get('muscleStrengthMetrics', {}).get('sitToStandLevel', 'no data available')})
    Context: Measures lower body strength and functional mobility. Important for daily activities and independence.

    • Movement Quality: {client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexion', 'no data available')} ({client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexionLevel', 'no data available')})
    Context: Assesses overall movement quality and joint mobility. Critical for injury prevention and functional movement."""

        # Format the complete prompt
        formatted_data = f"""
        CLIENT PROFILE:
        {profile}

        DETAILED FITNESS ASSESSMENT:
        {fitness}

        HEALTH SPAN ASSESSMENT:
        {health_span}
        """

        return {
            "system": cls.system,
            "user": cls.user.format(formatted_data=formatted_data)
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = structured_text_cost_parser(response, model)
        return response, cost


@PromptRegistry.register("healthy_signals_critic")
class HealthySignalsCriticPrompt:
    system = """You are an expert exercise physiologist specializing in health optimization. Your role is to analyze three different health domain target recommendations and synthesize them into a final recommendation that captures the best insights while addressing any inconsistencies.

Your task is to:
1. Review three different target score proposals
2. Identify common patterns and recommendations
3. Flag any concerning discrepancies or potential errors
4. Synthesize the recommendations into a final proposal that:
   - Maintains consistency with the client's profile
   - Incorporates the strongest reasoning from each proposal
   - Resolves any conflicts between recommendations
   - Ensures realistic and achievable targets

Remember that scores can be benchmarked as follows:
- Score of 100 represents meeting ACSM guidelines and basic health targets
- Score of 200 indicates progression beyond baseline recommendations
- Score of 300 represents excellence and peak performance
- Scoring should be scaled based on the client's Capacity to Exercise (CTE):
  * CTE < 50: Target scores typically between 50-150
  * CTE 50-75: Target scores typically between 100-200
  * CTE > 75: Target scores typically between 150-300

Provide your response in the following JSON format:
{
    "rationale": "Analysis of the proposals and explanation of final synthesis...",
    "display_text": "string",         // Detailed explanation without references to proposals
    "aerobic": float,                 // Target score (50-300)
    "bone": float,                    // Target score (50-300)
    "body_composition": float,        // Target score (50-300)
    "muscle_fitness": float,          // Target score (50-300)
    "movement_quality": float,        // Target score (50-300)
    "balance": float,                 // Target score (50-300)
    "cognitive_health": float         // Target score (50-300)
}"""

    user = """Review these three health domain target proposals for the client:

PROPOSAL 1:
{proposal_1}

PROPOSAL 2:
{proposal_2}

PROPOSAL 3:
{proposal_3}

CLIENT CONTEXT:
{formatted_data}

Analyze the proposals for:
1. Common patterns in target scores across domains
2. Significant discrepancies between recommendations
3. How well each aligns with the client's profile and current metrics
4. Potential errors or unrealistic targets

Synthesize these into a final recommendation that:
- Captures the most well-reasoned target scores
- Resolves any conflicts or inconsistencies
- Ensures alignment with client capacity and goals
- Maintains realistic and achievable targets
- Considers interdependencies between health domains
- Progress the clients in areas where they are deficient from the HEALTH SPAN ASSESSMENT

The score for any two targets should not be the same. You should pick integers continous between 50 and 300.

Please cite specific data from the client profile in your display_text, using the profile, survey questions, and actual raw assessment data as needed.
You should refer to specific answers, blood tests, fitness assessments, and other relevant data points in the display_text.
Provide your analysis and final recommendation in the specified JSON format."""

    class DataModel(BaseModel):
        rationale: str
        display_text: str
        aerobic: float
        bone: float
        body_composition: float
        muscle_fitness: float
        movement_quality: float
        balance: float
        cognitive_health: float

    @classmethod
    def format_prompt(cls, client_data: dict, proposals: list[dict]) -> dict:
        # Format personal profile (simplified for critic)
        profile = f"""The client is a {calculate_age(client_data.get("person", {}).get("date_of_birth", 'no data available'))}-year-old {client_data.get("person", {}).get("gender", 'no data available')}.

Current Health Metrics:
• Aerobic Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("aerobic_fitness", 'no data available')}/100
• Body Composition: {client_data.get("assessments", {}).get("healthSpan", {}).get("body_composition", 'no data available')}/100
• Muscular Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("muscular_fitness", 'no data available')}/100
• Joint Fitness: {client_data.get("assessments", {}).get("healthSpan", {}).get("joint_fitness", 'no data available')}/100
• Balance: {client_data.get("assessments", {}).get("healthSpan", {}).get("balance", 'no data available')}/100
• Cognitive Health: {client_data.get("assessments", {}).get("healthSpan", {}).get("cognitive_health", 'no data available')}/100

Key Fitness Indicators:
• VO2 Peak: {client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('value', 'no data available')} {client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('unit', 'no data available')} ({client_data.get('aerobicMetrics', {}).get('vo2Peak', {}).get('level', 'no data available')})
• Body Fat: {client_data.get('bodyCompositionMetrics', {}).get('percent_fat_value', 'no data available')}% ({client_data.get('bodyCompositionMetrics', {}).get('percent_fat_level', 'no data available')})
• Movement Quality: {client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexion', 'no data available')} ({client_data.get('movementQualityMetrics', {}).get('multiSegmentFlexionLevel', 'no data available')})

Exercise Capacity:
• Current Capacity to Exercise (CTE): {client_data.get("assessments", {}).get("lifestyle", {}).get("cte", 'no data available')}/100

Goals (in priority order):
1. {client_data.get("goals", {}).get("primary_goal", 'no data available')}
2. {client_data.get("goals", {}).get("secondary_goal", 'no data available')}
3. {client_data.get("goals", {}).get("tertiary_goal", 'no data available')}"""

        # Format the proposals
        proposal_texts = []
        for i, prop in enumerate(proposals[:3], 1):
            proposal_texts.append(f"""
Rationale: {prop.get('rationale', 'no data available')}
Aerobic Fitness: {prop.get('aerobic', 'no data available')}/100
Body Composition: {prop.get('body_composition', 'no data available')}/100
Muscular Fitness: {prop.get('muscle_fitness', 'no data available')}/100
Movement Quality: {prop.get('movement_quality', 'no data available')}/100
Bone Fitness: {prop.get('bone', 'no data available')}/100
Balance: {prop.get('balance', 'no data available')}/100
Cognitive Health: {prop.get('cognitive_health', 'no data available')}/100""")

        return {
            "system": cls.system,
            "user": cls.user.format(
                proposal_1=proposal_texts[0],
                proposal_2=proposal_texts[1],
                proposal_3=proposal_texts[2],
                formatted_data=profile
            )
        }

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        response, cost = structured_text_cost_parser(response, model)
        return response, cost

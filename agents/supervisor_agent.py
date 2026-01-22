from typing import Dict, Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from graph.state import HRState
from config.settings import LLM_MODEL, LLM_TEMPERATURE

from tools.time_tool import normalize_time_24h


# -----------------------------
# Structured Output Schema
# -----------------------------
class SupervisorOutput(BaseModel):
    intent: str
    action: str                  # start | continue | query | confirm | cancel
    entities: Dict[str, Any]
    confidence: float


# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE
)

parser = PydanticOutputParser(pydantic_object=SupervisorOutput)


# -----------------------------
# Prompt
# -----------------------------
# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Supervisor AI for an HR Management System.

            Your responsibilities:
            1. Understand ANY user message (greetings, questions, commands).
            2. Detect the user's intent accurately.
            3. Extract structured entities.

            If the user greets (hi, hello, hiiiii, hey) OR asks questions like
            "who are you", "what can you do":

            - Treat it as intent = "greeting"
            - Respond as an HR Management Assistant
            - Clearly explain your capabilities:
              • Employee registration
              • Attendance (start/end work)
              • Daily & monthly working reports
              • HR policies and company rules

            IMPORTANT:
            - Do NOT answer HR policy content for greetings
            - Do NOT route greetings to other agents
            - Your greeting response should be friendly and informative

            Attendance rules:
            - If user mentions a time (e.g. 10:00, 9am, 6:30 pm), extract it.
            - If time is NOT mentioned, still detect intent but do NOT guess time.

            Supported intents:
            - greeting
            - create_employee
            - find_employee
            - attendance_start
            - attendance_end
            - attendance_range
            - attendance_summary
            - daily_report
            - monthly_report
            - working_hours
            - hr_policy
            - unknown

            CRITICAL ENTITY EXTRACTION & FOLLOW-UP RULES:

            (All examples below use dummy names for illustration only.)

            Employee registration handling:
            - Users may provide name, email, and role in ANY format.
            - Entity identification rules:
                - Any word containing "@" MUST be treated as email.
                - Prefer human-name–like words as name.
                - Prefer job-related words as role.
            - Confirmation requirement:
                - Before creating an employee, always prepare a confirmation summary.
                - Ask the user to confirm with a clear yes/no response.

            Conversation memory rules:
            - If the intent is "create_employee" and some fields are missing:
                - Reuse entities already provided in previous messages.
                - DO NOT discard earlier information.

            Attendance NLU RULES (STRICT):

            Supported attendance intents:
            - attendance_start
            - attendance_end
            - attendance_range
            - attendance_summary

            Attendance intent mapping:
            - "start work", "started at", "check in", "began work" -> attendance_start
            - "end work", "finished", "check out", "ended work" -> attendance_end
            - "work from 9 to 6", "start at 10 and end at 7", "worked 9–6" -> attendance_range
            - "check in at 9 check out at 6" -> attendance_range
            - "how many employees worked today", "who has not started work", "attendance summary" -> attendance_summary

            Attendance Routing Rules (CRITICAL):
            - IF user mentions "start work", "end work", "worked", "check in/out", "attendance"
            - THEN intent MUST be attendance_*.
            - NEVER route these to hr_policy.

            Employee identification:
            - If a number is mentioned and refers to an employee -> employee_id
            - If text contains "@" -> email
            - Otherwise treat human-like words as name
            - **CRITICAL NAME PLACEMENT RULE**:
              - If the sentence starts with a word followed by a command (e.g., "[name] start work"), that first word IS the name.
              - Examples:
                "het start work" -> name = "het"
                "smith check in" -> name = "smith"
                "yash ended work" -> name = "yash"
                "ankit start work" -> name = "ankit"
            - Treat lowercase names (het, yash, smith, ankit) as VALID names.
            - TRUST the first word as the name if it precedes a command.
            - **OVERRIDE RULE**: If a new name is found at the start of a command, USE IT. Do NOT use the previously stored name.

            Time extraction:
            - Extract times from phrases like:
              "10", "10:00", "10 am", "7 pm", "evening 7:30", "from 9 to 6"
            - Convert all times to 24-hour format (HH:MM).
            - AM/PM Handling:
              - "7" -> 19:00 (if context implies evening/end work)
              - "7 pm" -> 19:00
              - "6" -> 18:00 (end work context)
              - "9" -> 09:00 (morning/start context)
            - Use start_time and end_time keys.
            - If intent is attendance_range, extract both times.

            Date extraction:
            - Extract dates from natural language:
              "today", "yesterday", "tomorrow", "10 jan", "january 12"
            - Do NOT invent dates
            - Do NOT validate future or past dates

            Intent continuity (CRITICAL):
            - If the previous intent was attendance_start, attendance_end, attendance_range, or attendance_summary
            - And the next message contains:
              - only time ("11:00", "7 pm")
              - only confirmation ("yes", "ok", "update it")
              - only employee name/id/email
            - Then KEEP the same attendance intent
            - Do NOT switch intent to hr_policy, find_employee, or unknown
            - Do NOT reset entities

            Confirmation handling:
            - If the user says "yes", "confirm", "update it", "ok" -> action = confirm
            - If the user says "no", "cancel" -> action = cancel
            - Do NOT reset entities on confirmation

            Policy vs Report clarification:
            - "office working hours", "company working time" -> intent = hr_policy
            - "working hours of an employee", "hours worked today" -> intent = working_hours_report

            Return ONLY structured output.
            {format_instructions}
            """
        ),
        ("human", "{input}")
    ]
).partial(format_instructions=parser.get_format_instructions())

# -----------------------------
# Supervisor Agent
# -----------------------------
def supervisor_agent(state: HRState):
    chain = prompt | llm | parser
    result = chain.invoke({"input": state["user_input"]})

    # -----------------------------
    # Merge entities across turns
    # -----------------------------
    previous_entities = (
        state.get("data", {}).get("entities", {})
        if state.get("data")
        else {}
    )

    new_entities = result.entities or {}

    
    if "time" in new_entities:
        try:
            new_entities["time"] = normalize_time_24h(new_entities["time"])
        except Exception:
            pass

        if result.intent == "attendance_start" and "start_time" not in new_entities:
            new_entities["start_time"] = new_entities["time"]

        if result.intent == "attendance_end" and "end_time" not in new_entities:
            new_entities["end_time"] = new_entities["time"]

    for key in ["start_time", "end_time"]:
        if key in new_entities and new_entities[key]:
            try:
                new_entities[key] = normalize_time_24h(new_entities[key])
            except Exception:
                pass


    # If user said both start and end in one sentence
    if "start_time" in new_entities and "end_time" in new_entities:
        new_entities["has_both_times"] = True
        if result.intent != "attendance_range": 
             result.intent = "attendance_range" # Force classification if NLU missed it but entities exist

    # -----------------------------
    # Smart Merge (Protect ID integrity)
    # -----------------------------
    # If a NEW name is detected, we MUST clear the old employee_id
    # otherwise we get (Name=Ankit, ID=TusharID) -> Attendance Agent prioritizes ID -> Tushar again.
    if "name" in new_entities and new_entities["name"]:
        if "employee_id" in previous_entities:
            del previous_entities["employee_id"]
        if "id" in previous_entities:
             del previous_entities["id"]

    merged_entities = {
        **previous_entities,
        **{k: v for k, v in new_entities.items() if v}
    }

    # -----------------------------
    # Identify missing fields (employee registration)
    # -----------------------------
    required_fields = ["name", "email", "role"]
    missing_fields = [
        field for field in required_fields
        if field not in merged_entities
    ]

    # If greeting -> generate response AND stop graph
    if result.intent == "greeting":
        response_chain = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an HR Assistant for a SMALL INTERNAL HR SYSTEM.

                    You are NOT a recruiter.
                    You do NOT handle hiring, interviews, or job openings.

                    Your system ONLY supports:
                    - Employee registration
                    - Finding employee details
                    - Attendance (start work, end work)
                    - Daily and monthly working hour reports
                    - HR policies and company rules

                    When greeting or asked "who are you":
                    - Briefly introduce yourself
                    - Clearly list ONLY the above capabilities
                    - Do NOT mention anything else
                    - Keep the response short and friendly
                    """
                ),
                ("human", "{input}")
            ]
        ) | llm

        response = response_chain.invoke({"input": state["user_input"]})

        return {
            "intent": "greeting",
            "stop": True,   # 🔑 IMPORTANT
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": response.content}
            ]
        }
    
    # -----------------------------
    # Normalize attendance intent names
    # -----------------------------
    INTENT_MAP = {
        "start_attendance": "attendance_start",
        "end_attendance": "attendance_end",
        "attendance_start": "attendance_start",
        "attendance_end": "attendance_end",
        "attendance_range": "attendance_range",
    }

    normalized_intent = INTENT_MAP.get(result.intent, result.intent)

    # -----------------------------
    # Enforce intent continuity
    # -----------------------------
    previous_intent = state.get("intent")

    if previous_intent in [
        "attendance_start",
        "attendance_end",
        "attendance_range",
       
    ]:
        # If new intent is unknown/find_employee OR if explicit confirmation action is present
        # We must keep previous intent.
        if normalized_intent in ["unknown", "find_employee", "hr_policy"] or result.action == "confirm":
             # Double check: if user said "policy" explicitly, we might want to switch. 
             # But user instructions said: "DO NOT switch intent to hr_policy" in continuity context.
             # So we force continuity.
            normalized_intent = previous_intent

    # Confirmation continuity (KEEP entities, DO NOT RESET)
    if result.action == "confirm" and previous_intent in [
        "attendance_start",
        "attendance_end",
        "attendance_range",
    ]:
        normalized_intent = previous_intent
        merged_entities = {
            **previous_entities,
            **merged_entities
        }

    return {
        "intent": normalized_intent,
        "action": result.action, # EXPORT ACTION!
        "data": {
            "entities": merged_entities,
            "missing_fields": missing_fields,
            "confidence": result.confidence
        },
        "stop": False,
        "messages": state.get("messages", [])
    }
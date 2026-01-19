from typing import Dict, Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from graph.state import HRState
from config.settings import LLM_MODEL, LLM_TEMPERATURE


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
            - start_attendance
            - end_attendance
            - daily_report
            - monthly_report
            - working_hours
            - hr_policy
            - unknown

            CRITICAL ENTITY EXTRACTION & FOLLOW-UP RULES:

            (All examples below use dummy names for illustration only.)

            Employee registration handling:
            - Users may provide name, email, and role in ANY format, including:
            • "alex, alex@example.com, QA"
            • "alex alex@example.com QA"
            • "my name is alex, email alex@example.com, role QA"
            • "alex, QA" (partial)
            • "alex" (partial)

            Entity identification rules:
            - Any word containing "@" MUST be treated as email.
            - Role and name do NOT have fixed positions.
            - Do NOT assume the first value is always the name.

            Comma- or space-separated input handling:
            - Identify email first using "@"
            - From remaining values:
            • If a value matches common role patterns (QA, AI, ML, HR, Dev, Engineer, Tester),
                treat it as role
            • Otherwise, treat it as name
            - If multiple non-email values exist:
            • Prefer human-name–like words as name
            • Prefer job-related words as role

            Short / ambiguous role handling:
            - Short or abbreviated role terms (case-insensitive) may be ambiguous, such as:
                - qa, q.a, quality
                - ai, artificial intelligence
                - ml, machine learning
                - dev, developer
                - tester, testing
                - powerbi, power bi

                Treat role detection as semantic, not exact-match.
            - Expand them to reasonable full forms when possible:
            • "QA" → "QA Engineer"
            • "AI" → "AI Developer"
            • "ML" → "ML Engineer"
            - If expansion is uncertain, keep the original value

            Confirmation requirement:
            - Before creating an employee, always prepare a confirmation summary:
            Name, Email, Role
            - Ask the user to confirm with a clear yes/no response.
            - Examples of confirmation:
            • "Yes"
            • "Confirm"
            • "Looks good"
            - If the user says NO or corrects details:
            - Update the entities
            - Show the confirmation again
            - If the user denies registration:
            - Do NOT proceed

            Partial input rules:
            - If only ONE value is provided:
            • Treat it as name
            - If TWO values are provided:
            • If one contains "@", treat as name + email
            • Otherwise, treat as name + role (tentative)

            Conversation memory rules:
            - If the intent is "create_employee" and some fields are missing:
            • Reuse entities already provided in previous messages
            • Merge new entities with previously extracted entities
            • DO NOT discard earlier information

            PARTIAL REGISTRATION HANDLING:

            - If intent is "create_employee" and some fields are already extracted:
            • Identify exactly which fields are missing (name, email, role)
            • Do NOT treat missing fields as all fields missing
            • Do NOT reset previously extracted information
            - Clearly indicate which fields are missing so the next agent can ask ONLY for those fields

            IMPORTANT:
            - NEVER ignore user-provided information
            - NEVER assume extracted data is final without confirmation
            - Always extract the BEST POSSIBLE entities from the input

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

    # If greeting → generate response AND stop graph
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

    return {
        "intent": result.intent,
        "data": {
            "entities": merged_entities,
            "missing_fields": missing_fields,
            "confidence": result.confidence
        },
        "stop": False,
        "messages": state.get("messages", [])
    }
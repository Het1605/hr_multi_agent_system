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
            You are a Supervisor AI for an HR management system.

            Your task:
            1. Understand the user's message (any language, spelling, greetings).
            2. Detect the correct intent.
            3. Extract entities if present.

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

            Entity examples:
            - name
            - email
            - role
            - date
            - month
            - year

            If the message is only a greeting, intent = "greeting".
            If unsure, intent = "unknown".

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
def supervisor_agent(state: HRState) -> Dict:
    chain = prompt | llm | parser

    result: SupervisorOutput = chain.invoke(
        {"input": state["user_input"]}
    )

    return {
        "intent": result.intent,
        "data": {
            "entities": result.entities,
            "confidence": result.confidence
        },
        "messages": state.get("messages", []) + [
            {
                "role": "system",
                "content": f"Intent detected: {result.intent}"
            }
        ]
    }
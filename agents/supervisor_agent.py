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
            "entities": result.entities,
            "confidence": result.confidence
        },
        "stop": False,
        "messages": state.get("messages", [])
    }
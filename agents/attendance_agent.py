from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import HRState
from tools.db_tool import (
    get_employee_by_id,
    get_employee_by_email,
    get_employees_by_name,
    get_attendance_for_employee_on_date,
    start_attendance,
    end_attendance,
    get_attendance_summary_for_date,
)
from tools.time_tool import (
    current_date,
    normalize_natural_date,
    is_future_date,
)
from config.settings import LLM_MODEL, LLM_TEMPERATURE


llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# -----------------------------
# LLM PROMPT (POLISH ONLY)
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an Attendance Assistant.

            Your job is ONLY to rephrase messages clearly and politely.

            You will be given a message describing the result of an attendance operation
            or a request for missing information.

            RULES:
            - Do NOT add new information
            - Do NOT remove information
            - Do NOT ask new questions
            - Do NOT change intent
            - Do NOT explain internal logic
            - Do NOT mention HR policies
            - Do NOT infer anything

            Only rewrite the given message in clear, professional English.
            """
        ),
        ("human", "{input}")
    ]
)


def _reply(state: HRState, text: str) -> Dict:
    """Utility: polish text with LLM"""
    chain = prompt | llm
    response = chain.invoke({"input": text})
    return {
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": response.content}
        ]
    }


# -----------------------------
# ATTENDANCE AGENT
# -----------------------------
def attendance_agent(state: HRState) -> Dict:
    intent = state.get("intent")
    action = state.get("action")
    entities = state.get("data", {}).get("entities", {})

    # -----------------------------
    # DATE HANDLING
    # -----------------------------
    if "date" in entities:
        attendance_date = normalize_natural_date(entities["date"])
    else:
        attendance_date = current_date()

    if is_future_date(attendance_date):
        return _reply(
            state,
            "You cannot assign attendance for a future date."
        )

    # -----------------------------
    # SUMMARY
    # -----------------------------
    if intent == "attendance_summary":
        summary = get_attendance_summary_for_date(attendance_date)
        return _reply(
            state,
            f"On {attendance_date}, {summary['present']} employees worked "
            f"and {summary['absent']} employees did not start work."
        )

    # -----------------------------
    # RESOLVE EMPLOYEE
    # -----------------------------
    employee = None

    emp_id = entities.get("employee_id") or entities.get("id")
    if emp_id:
        employee = get_employee_by_id(int(emp_id))

    elif "email" in entities:
        employee = get_employee_by_email(entities["email"])

    elif "name" in entities:
        matches = get_employees_by_name(entities["name"])
        if len(matches) == 1:
            employee = matches[0]
        elif len(matches) > 1:
            return _reply(
                state,
                "Multiple employees found with this name. "
                "Please provide the employee ID."
            )
        else:
            return _reply(state, "No employee found with this name.")

    if not employee:
        return _reply(
            state,
            "Please provide the employee name or ID."
        )

    emp_id = employee["id"]
    name = employee["name"]

    start_time = entities.get("start_time")
    end_time = entities.get("end_time")

    # -----------------------------
    # START ATTENDANCE
    # -----------------------------
    if intent == "attendance_start":
        if not start_time:
            return _reply(
                state,
                f"Please provide the start time for {name}."
            )

        existing = get_attendance_for_employee_on_date(emp_id, attendance_date)

        if existing and existing.get("start_time") and action != "confirm":
            return _reply(
                state,
                f"{name} already has a start time for {attendance_date}. "
                "Do you want to update it?"
            )

        start_attendance(emp_id, attendance_date, start_time)
        return _reply(
            state,
            f"Work started for {name} at {start_time} on {attendance_date}."
        )

    # -----------------------------
    # END ATTENDANCE
    # -----------------------------
    if intent == "attendance_end":
        if not end_time:
            return _reply(
                state,
                f"Please provide the end time for {name}."
            )

        existing = get_attendance_for_employee_on_date(emp_id, attendance_date)

        if not existing or not existing.get("start_time"):
            return _reply(
                state,
                f"{name} has not started work yet on {attendance_date}."
            )

        if existing.get("end_time") and action != "confirm":
            return _reply(
                state,
                f"{name} already has an end time for {attendance_date}. "
                "Do you want to update it?"
            )

        end_attendance(emp_id, attendance_date, end_time)
        return _reply(
            state,
            f"Work ended for {name} at {end_time} on {attendance_date}."
        )

    return _reply(state, "I could not process this attendance request.")
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
)
from tools.time_tool import current_date, current_time
from config.settings import LLM_MODEL, LLM_TEMPERATURE


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an Attendance Management Agent.

            Your responsibilities:
            - Handle employee check-in and check-out
            - Accept user-provided time if available
            - Prevent invalid attendance operations
            - Never assume employee name is unique
            """
        ),
        ("human", "{input}")
    ]
)


def attendance_agent(state: HRState) -> Dict:
    intent = state["intent"]
    entities = state.get("data", {}).get("entities", {})
    user_input = state["user_input"]

    today = current_date()

    response_text = ""

    # -----------------------------
    # RESOLVE EMPLOYEE
    # -----------------------------
    employee = None

    if "email" in entities:
        employee = get_employee_by_email(entities["email"])

    elif "id" in entities:
        employee = get_employee_by_id(int(entities["id"]))

    elif "name" in entities:
        matches = get_employees_by_name(entities["name"])
        if len(matches) == 1:
            employee = matches[0]
        elif len(matches) > 1:
            response_text = (
                "Multiple employees found with this name. "
                "Please specify the employee ID:\n"
            )
            for e in matches:
                response_text += f"- ID {e['id']}: {e['name']} ({e['role']})\n"
        else:
            response_text = "No employee found with this name."

    else:
        response_text = "Please specify the employee (name, email, or ID)."

    if response_text:
        return {
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": response_text}
            ]
        }

    emp_id = employee["id"]

    # -----------------------------
    # START ATTENDANCE
    # -----------------------------
    if intent == "start_attendance":
        start_time = entities.get("start_time") or current_time()
        existing = get_attendance_for_employee_on_date(emp_id, today)

        if existing and existing["start_time"]:
            response_text = "Work has already been started for today."
        else:
            start_attendance(emp_id, today, start_time)
            response_text = (
                f"Work started for {employee['name']} at {start_time}."
            )

    # -----------------------------
    # END ATTENDANCE
    # -----------------------------
    elif intent == "end_attendance":
        end_time = entities.get("end_time") or current_time()
        existing = get_attendance_for_employee_on_date(emp_id, today)

        if not existing or not existing["start_time"]:
            response_text = "Work has not been started yet today."
        elif existing["end_time"]:
            response_text = "Work has already been ended for today."
        else:
            end_attendance(emp_id, today, end_time)
            response_text = (
                f"Work ended for {employee['name']} at {end_time}."
            )

    else:
        response_text = "Attendance agent could not handle this request."

    # -----------------------------
    # LLM RESPONSE POLISHING
    # -----------------------------
    chain = prompt | llm
    final_response = chain.invoke({"input": response_text})

    return {
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": final_response.content}
        ]
    }
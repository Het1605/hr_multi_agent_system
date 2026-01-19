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
from tools.time_tool import current_date
from config.settings import LLM_MODEL, LLM_TEMPERATURE


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE
)

# -----------------------------
# PROMPT (LANGUAGE ONLY)
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an Attendance Management Agent.

            You receive structured context describing:
            - intent
            - employee information
            - missing fields
            - attendance operation result

            RULES:
            - Never guess time or date
            - Ask only for missing information
            - Never mention internal state or fields
            - Keep responses short and clear
            - Do NOT answer HR policy questions
            """
        ),
        ("human", "{input}")
    ]
)

# -----------------------------
# ATTENDANCE AGENT
# -----------------------------
def attendance_agent(state: HRState) -> Dict:
    intent = state.get("intent")
    data = state.get("data", {})
    entities = data.get("entities", {})

    today = current_date()

    response_context = {
        "intent": intent,
        "entities": entities
    }

    # -----------------------------
    # RESOLVE EMPLOYEE (DETERMINISTIC)
    # -----------------------------
    employee = None

    if "employee_id" in entities:
        employee = get_employee_by_id(int(entities["employee_id"]))

    elif "email" in entities:
        employee = get_employee_by_email(entities["email"])

    elif "name" in entities:
        matches = get_employees_by_name(entities["name"])

        if len(matches) == 1:
            employee = matches[0]
        elif len(matches) > 1:
            response_context["result"] = "multiple_employees"
            response_context["employees"] = matches

            chain = prompt | llm
            final_response = chain.invoke({"input": response_context})

            return {
                "data": data,
                "messages": state.get("messages", []) + [
                    {"role": "assistant", "content": final_response.content}
                ]
            }

    # -----------------------------
    # RECOMPUTE REQUIRED FIELDS
    # -----------------------------
    required_fields = []

    if intent in ["attendance_start", "attendance_end"]:
        required_fields = ["employee", "time"]

    missing_fields = []

    if not employee:
        missing_fields.append("employee")

    if intent == "attendance_start":
        if not entities.get("start_time"):
            missing_fields.append("start_time")

    if intent == "attendance_end":
        if not entities.get("end_time"):
            missing_fields.append("end_time")

    # -----------------------------
    # ASK FOR MISSING INFO (NO LOGIC IN LLM)
    # -----------------------------
    if missing_fields:
        response_context["missing_fields"] = missing_fields

        chain = prompt | llm
        final_response = chain.invoke({"input": response_context})

        return {
            "data": data,
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": final_response.content}
            ]
        }

    emp_id = employee["id"]

    # -----------------------------
    # START ATTENDANCE
    # -----------------------------
    if intent == "attendance_start":
        start_time_value = entities["start_time"]
        existing = get_attendance_for_employee_on_date(emp_id, today)

        if existing and existing.get("start_time"):
            response_context["result"] = "already_started"
        else:
            start_attendance(emp_id, today, start_time_value)
            response_context["result"] = "start_recorded"
            response_context["start_time"] = start_time_value

    # -----------------------------
    # END ATTENDANCE
    # -----------------------------
    elif intent == "attendance_end":
        end_time_value = entities["end_time"]
        existing = get_attendance_for_employee_on_date(emp_id, today)

        if not existing or not existing.get("start_time"):
            response_context["result"] = "not_started"
        elif existing.get("end_time"):
            response_context["result"] = "already_ended"
        else:
            end_attendance(emp_id, today, end_time_value)
            response_context["result"] = "end_recorded"
            response_context["end_time"] = end_time_value

    # -----------------------------
    # FINAL RESPONSE (LLM PHRASES ONLY)
    # -----------------------------
    chain = prompt | llm
    final_response = chain.invoke({"input": response_context})

    return {
        "data": data,
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": final_response.content}
        ]
    }
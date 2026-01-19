from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import HRState
from tools.db_tool import (
    create_employee,
    get_employee_by_email,
    get_employee_by_id,
    get_employees_by_name,
    get_employees_by_role,
    get_all_employees,
)
from config.settings import LLM_MODEL, LLM_TEMPERATURE


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE
)


# -----------------------------
# PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an Employee Management Agent in an HR system.

            Your responsibilities:
            - Create new employees
            - Find employee details
            - Handle duplicate names safely
            - Ask clarification if needed
            - Respond clearly to the user

            Never assume uniqueness of name or role.
            """
        ),
        ("human", "{input}")
    ]
)


# -----------------------------
# EMPLOYEE AGENT
# -----------------------------
def employee_agent(state: HRState) -> Dict:
    intent = state["intent"]
    entities = state.get("data", {}).get("entities", {})
    user_input = state["user_input"]

    response_text = ""

    # -----------------------------
    # CREATE EMPLOYEE
    # -----------------------------
    if intent == "create_employee":
        name = entities.get("name")
        email = entities.get("email")
        role = entities.get("role")

        if not name or not email or not role:
            response_text = (
                "To register an employee, I need name, email, and role."
            )
        else:
            existing = get_employee_by_email(email)
            if existing:
                response_text = (
                    f"An employee with email {email} already exists."
                )
            else:
                emp_id = create_employee(name, email, role)
                response_text = (
                    f"Employee {name} registered successfully with ID {emp_id}."
                )

    # -----------------------------
    # FIND EMPLOYEE
    # -----------------------------
    elif intent == "find_employee":
        if "email" in entities:
            employee = get_employee_by_email(entities["email"])
            employees = [employee] if employee else []

        elif "id" in entities:
            employee = get_employee_by_id(int(entities["id"]))
            employees = [employee] if employee else []

        elif "name" in entities:
            employees = get_employees_by_name(entities["name"])

        elif "role" in entities:
            employees = get_employees_by_role(entities["role"])

        else:
            employees = get_all_employees()

        if not employees:
            response_text = "No employee found."

        elif len(employees) == 1:
            e = employees[0]
            response_text = (
                f"Employee found:\n"
                f"Name: {e['name']}\n"
                f"Email: {e['email']}\n"
                f"Role: {e['role']}\n"
                f"ID: {e['id']}"
            )

        else:
            response_text = (
                "Multiple employees found. Please specify one:\n"
            )
            for e in employees:
                response_text += (
                    f"- ID {e['id']}: {e['name']} ({e['role']})\n"
                )

    else:
        response_text = "Employee agent could not handle this request."

    # -----------------------------
    # LLM RESPONSE POLISHING
    # -----------------------------
    chain = prompt | llm
    final_response = chain.invoke({"input": response_text})

    return {
        "data": state.get("data", {}),
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": final_response.content}
        ]
    }
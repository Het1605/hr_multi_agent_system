from typing import Dict
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
    intent = state.get("intent")
    data = state.get("data", {})
    entities = data.get("entities", {})
    
    # Recompute missing fields deterministically (SOURCE OF TRUTH)
    required_fields = ["name", "email", "role"]

    missing_fields = [
        field for field in required_fields
        if not entities.get(field)
    ]

    response_context = {
        "intent": intent,
        "entities": entities,
        "missing_fields": missing_fields
    }

    # -----------------------------
    # CREATE EMPLOYEE
    # -----------------------------
    if intent == "create_employee":
        # If something is missing, let LLM ask properly
        safe_missing_fields = [
            f for f in missing_fields if f not in entities
        ]

        if safe_missing_fields:
            response_context["missing_fields"] = safe_missing_fields

            chain = prompt | llm
            final_response = chain.invoke({"input": response_context})

            return {
                "data": data,
                "messages": state.get("messages", []) + [
                    {"role": "assistant", "content": final_response.content}
                ]
            }

        # All details present → create employee
        name = entities.get("name")
        email = entities.get("email")
        role = entities.get("role")

        existing = get_employee_by_email(email)
        if existing:
            response_context["result"] = "duplicate_email"
        else:
            emp_id = create_employee(name, email, role)
            response_context["result"] = "created"
            response_context["employee_id"] = emp_id

        chain = prompt | llm
        final_response = chain.invoke({"input": response_context})

        return {
            "data": data,
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": final_response.content}
            ]
        }
    
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

        response_context["employees"] = employees

        chain = prompt | llm
        final_response = chain.invoke({"input": response_context})

        return {
            "data": data,
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": final_response.content}
            ]
        }
    

    if intent == "employee_find_all":
        employees = get_all_employees()

        if not employees:
            return {
                "messages": state.get("messages", []) + [
                    {"role": "assistant", "content": "No employees found."}
                ]
            }

        lines = ["Here are all employees:"]
        for e in employees:
            lines.append(
                f"- ID {e['id']}: {e['name']} ({e['role']}, {e['email']})"
            )

        return {
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": "\n".join(lines)}
            ]
        }

    # -----------------------------
    # FALLBACK
    # -----------------------------
    chain = prompt | llm
    final_response = chain.invoke({"input": response_context})

    return {
        "data": data,
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": final_response.content}
        ]
    }
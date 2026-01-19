from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import HRState
from tools.db_tool import (
    get_employee_by_id,
    get_employee_by_email,
    get_employees_by_name,
    get_attendance_for_employee_on_date,
    get_attendance_for_employee,
)
from tools.time_tool import (
    calculate_duration_hours,
    month_date_range,
)
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
            You are a Report Agent in an HR system.

            Your responsibilities:
            - Generate daily and monthly work reports
            - Calculate working hours accurately
            - Handle missing or incomplete attendance data
            - Respond clearly to users
            """
        ),
        ("human", "{input}")
    ]
)


def report_agent(state: HRState) -> Dict:
    intent = state["intent"]
    entities = state.get("data", {}).get("entities", {})
    user_input = state["user_input"]

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
    # DAILY REPORT
    # -----------------------------
    if intent in ["daily_report", "working_hours"]:
        date = entities.get("date")

        if not date:
            response_text = "Please specify the date for the report."
        else:
            attendance = get_attendance_for_employee_on_date(emp_id, date)

            if not attendance or not attendance["start_time"]:
                response_text = f"No attendance found for {employee['name']} on {date}."
            elif not attendance["end_time"]:
                response_text = (
                    f"{employee['name']} started work at {attendance['start_time']} "
                    f"on {date}, but has not ended work yet."
                )
            else:
                hours = calculate_duration_hours(
                    attendance["start_time"],
                    attendance["end_time"]
                )
                response_text = (
                    f"{employee['name']} worked {hours} hours on {date}."
                )

    # -----------------------------
    # MONTHLY REPORT
    # -----------------------------
    elif intent == "monthly_report":
        month = entities.get("month")
        year = entities.get("year")

        if not month or not year:
            response_text = "Please specify both month and year for the report."
        else:
            start_date, end_date = month_date_range(int(year), int(month))
            records = get_attendance_for_employee(emp_id)

            total_hours = 0.0
            daily_summary: List[str] = []

            for r in records:
                if start_date <= r["date"] <= end_date:
                    if r["start_time"] and r["end_time"]:
                        hours = calculate_duration_hours(
                            r["start_time"],
                            r["end_time"]
                        )
                        total_hours += hours
                        daily_summary.append(
                            f"{r['date']}: {hours} hours"
                        )
                    else:
                        daily_summary.append(
                            f"{r['date']}: incomplete attendance"
                        )

            if not daily_summary:
                response_text = (
                    f"No attendance records found for {employee['name']} "
                    f"in {month}/{year}."
                )
            else:
                response_text = (
                    f"Monthly working report for {employee['name']} ({month}/{year}):\n"
                    + "\n".join(daily_summary)
                    + f"\n\nTotal hours: {round(total_hours, 2)}"
                )

    else:
        response_text = "Report agent could not handle this request."

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
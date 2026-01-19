from typing import Dict
from graph.state import HRState


def supervisor_agent(state: HRState) -> Dict:
    user_input = state["user_input"].lower()

    intent = None

    # -----------------------
    # EMPLOYEE INTENTS
    # -----------------------
    if any(word in user_input for word in ["register", "add employee", "create employee"]):
        intent = "create_employee"

    elif any(word in user_input for word in ["find employee", "employee detail", "show employee"]):
        intent = "find_employee"

    # -----------------------
    # ATTENDANCE INTENTS
    # -----------------------
    elif any(word in user_input for word in ["start work", "start day", "check in"]):
        intent = "start_attendance"

    elif any(word in user_input for word in ["end work", "end day", "check out"]):
        intent = "end_attendance"

    # -----------------------
    # REPORT INTENTS
    # -----------------------
    elif "monthly" in user_input and any(word in user_input for word in ["record", "report", "working"]):
        intent = "monthly_report"

    elif "daily" in user_input and any(word in user_input for word in ["report", "attendance"]):
        intent = "daily_report"

    elif any(word in user_input for word in ["working hours", "how many hours"]):
        intent = "working_hours"

    # -----------------------
    # KNOWLEDGE / POLICY
    # -----------------------
    elif any(word in user_input for word in ["policy", "rule", "leave", "hr"]):
        intent = "hr_policy"

    # -----------------------
    # FALLBACK
    # -----------------------
    else:
        intent = "unknown"

    return {
        "intent": intent,
        "messages": state.get("messages", []) + [
            {
                "role": "system",
                "content": f"Detected intent: {intent}"
            }
        ]
    }
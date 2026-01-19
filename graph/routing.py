from graph.state import HRState


def route_by_intent(state: HRState) -> str:
    intent = state.get("intent")

    if intent in ["create_employee", "find_employee"]:
        return "employee_agent"

    if intent in ["start_attendance", "end_attendance"]:
        return "attendance_agent"

    if intent in ["daily_report", "monthly_report", "working_hours"]:
        return "report_agent"

    if intent == "hr_policy":
        return "knowledge_agent"


    return "knowledge_agent"
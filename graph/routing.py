from graph.state import HRState


def route_by_intent(state: HRState) -> str:
    """
    Decide which agent should handle the request
    based on the intent decided by the supervisor.
    """

    intent = state.get("intent")

    # -----------------------------
    # Employee-related intents
    # -----------------------------
    if intent in [
        "create_employee",
        "find_employee",
        "employee_find_all",
        "employee_find_last",
        "employee_find_by_role",
        "employee_find_by_name",
    ]:
        return "employee_agent"

    # -----------------------------
    # Attendance-related intents
    # -----------------------------
    if intent in [
        "attendance_start",
        "attendance_end",
        "attendance_daily_report",
        "attendance_monthly_report",
        "attendance_working_hours",
    ]:
        return "attendance_agent"

    # -----------------------------
    # Report-related intents
    # -----------------------------
    if intent in [
        "daily_report",
        "monthly_report",
        "working_hours",
    ]:
        return "report_agent"

    # -----------------------------
    # Knowledge / policy
    # -----------------------------
    if intent == "hr_policy":
        return "knowledge_agent"

    # -----------------------------
    # Fallback (safe)
    # -----------------------------
    return "supervisor_agent"
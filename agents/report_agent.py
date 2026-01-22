from typing import Dict, List, Optional
from datetime import datetime
import calendar

from graph.state import HRState
from tools.db_tool import (
    get_employee_by_id,
    get_employee_by_email,
    get_employees_by_name,
    get_attendance_for_employee_on_date,
    get_attendance_for_employee,
    get_attendance_summary_for_date,
)
from tools.time_tool import (
    calculate_duration_hours,
    current_date,
    month_date_range,
    normalize_natural_date,
)


def format_date_verbose(date_str: str) -> str:
    """
    Convert YYYY-MM-DD to 'January 20, 2026'
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %d, %Y")
    except ValueError:
        return date_str


def format_attendance_summary(date_str: str, summary: Dict) -> str:
    """
    STRICT RULE:
    "On <Date>, <worked> employees worked and <not_worked> did not start work."
    Example: "On January 20, 2026, 4 employees worked and 6 did not start work."
    """
    verbose_date = format_date_verbose(date_str)
    worked = summary["worked"]
    not_worked = summary["not_worked"]
    
    # We follow the explicit example structure from the latest request
    return (
        f"On {verbose_date}, {worked} employees worked and "
        f"{not_worked} did not start work."
    )


def format_daily_report(name: str, date_str: str, attendance: Dict) -> str:
    """
    STRICT RULE:
    - Start + End: "{Name} worked X hours on <Date>."
    - Start only: "{Name} started work at <Start> on <Date>, but has not ended work yet."
    - No record: "{Name} did not work on <Date>."
    """
    verbose_date = format_date_verbose(date_str)
    
    if not attendance or not attendance.get("start_time"):
        return f"{name} did not work on {verbose_date}."
    
    start = attendance["start_time"]
    end = attendance.get("end_time")

    # Start only
    if start and not end:
        return (
            f"{name} started work at {start} on {verbose_date}, "
            f"but has not ended work yet."
        )

    # Start + End
    hours = calculate_duration_hours(start, end)
    # Format hours nicely (remove .0)
    hours_str = f"{int(hours)}" if hours % 1 == 0 else f"{hours}"
    
    return f"{name} worked {hours_str} hours on {verbose_date}."


def format_monthly_report(name: str, month: int, year: int, records: List[Dict]) -> str:
    """
    STRICT RULE:
    "Monthly working report for <Employee> (<Month Year>):"
    • <Date formated>: <Hours> hours / incomplete attendance
    Total hours worked: <Total>
    """
    start_date, end_date = month_date_range(year, month)
    month_name = calendar.month_name[month]
    
    header = f"Monthly working report for {name} ({month_name} {year}):"
    
    relevant_records = [
        r for r in records 
        if start_date <= r["date"] <= end_date 
        and (r.get("start_time") or r.get("end_time"))
    ]
    relevant_records.sort(key=lambda x: x["date"])

    if not relevant_records:
        return f"No attendance records found for {name} in {month_name} {year}."

    lines = [header]
    total_hours = 0.0

    for r in relevant_records:
        # Date format for list items: "Jan 10"? 
        # The prompt for Monthly Report says "One clear answer per response... No bullet points unless listing days in monthly report"
        # and Example: "• Jan 10: 8 hours"
        d_dt = datetime.strptime(r["date"], "%Y-%m-%d")
        day_str = d_dt.strftime("%b %d")

        if r.get("start_time") and r.get("end_time"):
            h = calculate_duration_hours(r["start_time"], r["end_time"])
            total_hours += h
            h_str = f"{int(h)}" if h % 1 == 0 else f"{h}"
            lines.append(f"• {day_str}: {h_str} hours")
        else:
            lines.append(f"• {day_str}: incomplete attendance")

    total_str = f"{int(total_hours)}" if total_hours % 1 == 0 else f"{total_hours}"
    lines.append(f"\nTotal hours worked: {total_str}")

    return "\n".join(lines)


def report_agent(state: HRState) -> Dict:
    intent = state.get("intent")
    entities = state.get("data", {}).get("entities", {})

    response_text = ""

    # =========================================================
    # ATTENDANCE SUMMARY
    # =========================================================
    if intent == "attendance_summary":
        raw_date = entities.get("date")
        date_str = normalize_natural_date(raw_date) if raw_date else current_date()
        if not date_str:
            date_str = current_date() # Fallback if normalization fails

        summary = get_attendance_summary_for_date(date_str)
        response_text = format_attendance_summary(date_str, summary)
        
        return {
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": response_text}
            ]
        }

    # =========================================================
    # EMPLOYEE RESOLUTION
    # =========================================================
    employee = None
    
    # Try different ID fields
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
            # FAIL-SAFE: Ambiguous
            return {
                "messages": state.get("messages", []) + [
                    {"role": "assistant", "content": "Employee information is ambiguous. Please provide ID."}
                ]
            }
        else:
             # FAIL-SAFE: Not found (handled below if employee stays None)
             pass

    # For employee reports, we MUST have an employee
    if not employee:
        # If we cannot resolve, return strict error
        return {
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": "Employee information is missing or ambiguous."}
            ]
        }

    emp_name = employee["name"]
    emp_db_id = employee["id"]

    # =========================================================
    # DAILY REPORT / WORKING HOURS
    # =========================================================
    if intent in ["daily_report", "working_hours", "working_hours_report"]:
        raw_date = entities.get("date")
        date_str = normalize_natural_date(raw_date) if raw_date else current_date()
        if not date_str:
             date_str = current_date()

        attendance = get_attendance_for_employee_on_date(emp_db_id, date_str)
        response_text = format_daily_report(emp_name, date_str, attendance)

    # =========================================================
    # MONTHLY REPORT
    # =========================================================
    elif intent == "monthly_report":
        # Determine month/year
        target_month = entities.get("month")
        target_year = entities.get("year")
        
        # Try to parse from date if present
        if "date" in entities and (not target_month or not target_year):
             iso_d = normalize_natural_date(entities["date"])
             if iso_d:
                 dt = datetime.strptime(iso_d, "%Y-%m-%d")
                 target_month = dt.month
                 target_year = dt.year

        now = datetime.now()
        if not target_month: target_month = now.month
        if not target_year: target_year = now.year

        records = get_attendance_for_employee(emp_db_id)
        response_text = format_monthly_report(emp_name, int(target_month), int(target_year), records)

    else:
        # Fallback for unknown intent routed here
        response_text = "Report type not supported."

    return {
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": response_text}
        ]
    }
from datetime import datetime, date, time, timedelta
from typing import Tuple, Optional
import re
import calendar


# -----------------------------
# EXISTING FUNCTIONS (UNCHANGED)
# -----------------------------
def current_date() -> str:
    return date.today().isoformat()


def current_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def current_datetime() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def parse_time(time_str: str) -> time:
    return datetime.strptime(time_str, "%H:%M:%S").time()


def month_date_range(year: int, month: int) -> Tuple[str, str]:
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)
    return start_date.isoformat(), end_date.isoformat()


def calculate_duration_hours(start_time: str, end_time: str) -> float:
    start = parse_time(start_time)
    end = parse_time(end_time)
    delta = datetime.combine(date.today(), end) - datetime.combine(date.today(), start)
    return round(delta.total_seconds() / 3600, 2)


# -----------------------------
# NEW ADDITIONS (ATTENDANCE)
# -----------------------------

_MONTH_MAP = {
    name.lower(): idx
    for idx, name in enumerate(calendar.month_name) if name
}
_MONTH_MAP.update({
    name.lower(): idx
    for idx, name in enumerate(calendar.month_abbr) if name
})


def normalize_natural_date(text: str) -> Optional[str]:
    """
    Convert natural language date to ISO format (YYYY-MM-DD).

    Supported:
    - today
    - yesterday
    - 10 jan / jan 10
    - 12 january / january 12

    Returns ISO date string or None if not detected.
    """

    text = text.lower().strip()
    today = date.today()

    if text == "today":
        return today.isoformat()

    if text == "yesterday":
        return (today - timedelta(days=1)).isoformat()

    # Match: 10 jan / 10 january
    match1 = re.search(r"(\d{1,2})\s+([a-zA-Z]+)", text)
    # Match: jan 10 / january 10
    match2 = re.search(r"([a-zA-Z]+)\s+(\d{1,2})", text)

    day = None
    month = None

    if match1:
        day = int(match1.group(1))
        month = _MONTH_MAP.get(match1.group(2).lower())

    elif match2:
        day = int(match2.group(2))
        month = _MONTH_MAP.get(match2.group(1).lower())

    if day and month:
        year = today.year
        try:
            parsed = date(year, month, day)
            return parsed.isoformat()
        except ValueError:
            return None

    return None


def is_future_date(date_str: str) -> bool:
    """
    Check if given ISO date is in the future.
    """
    given = parse_date(date_str)
    return given > date.today()


def is_past_date(date_str: str) -> bool:
    """
    Check if given ISO date is in the past.
    """
    given = parse_date(date_str)
    return given < date.today()




def normalize_time_24h(time_str: str) -> str:
    """
    Normalize any human time input to strict HH:MM (24-hour).
    Examples:
    - "7" → "19:00" (assumed evening for end_time)
    - "7 am" → "07:00"
    - "7 pm" → "19:00"
    - "19:30" → "19:30"
    - "9:00 AM" → "09:00"
    """

    if not time_str:
        raise ValueError("Empty time")

    t = time_str.strip().lower()

    # If only a number like "7" or "9"
    if re.fullmatch(r"\d{1,2}", t):
        hour = int(t)
        # IMPORTANT ASSUMPTION:
        # Bare numbers default to EVENING for end_time use-cases
        if hour <= 8:
            hour += 12
        return f"{hour:02d}:00"

    # If number:number without am/pm
    if re.fullmatch(r"\d{1,2}:\d{2}", t):
        hour, minute = map(int, t.split(":"))
        return f"{hour:02d}:{minute:02d}"

    # Handle am/pm formats
    try:
        dt = datetime.strptime(t.replace(".", ""), "%I:%M %p")
        return dt.strftime("%H:%M")
    except ValueError:
        pass

    try:
        dt = datetime.strptime(t.replace(".", ""), "%I %p")
        return dt.strftime("%H:%M")
    except ValueError:
        pass

    raise ValueError(f"Unrecognized time format: {time_str}")
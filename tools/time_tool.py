from datetime import datetime, date, time, timedelta
from typing import Tuple


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
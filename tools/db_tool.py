import sqlite3
from typing import Optional, List, Dict
from config.settings import DATABASE_PATH


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


# =========================
# EMPLOYEE OPERATIONS
# =========================

def create_employee(name: str, email: str, role: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO employees (name, email, role, created_at)
        VALUES (?, ?, ?, datetime('now'))
        """,
        (name, email, role),
    )

    conn.commit()
    employee_id = cursor.lastrowid
    conn.close()
    return employee_id


def get_employee_by_id(employee_id: int) -> Optional[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, email, role FROM employees WHERE id = ?",
        (employee_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "role": row[3],
    }


def get_employee_by_email(email: str) -> Optional[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, email, role FROM employees WHERE email = ?",
        (email,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "role": row[3],
    }


def get_employees_by_name(name: str) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, name, email, role
        FROM employees
        WHERE LOWER(name) = LOWER(?)
        """,
        (name,),
    )

    rows = cursor.fetchall()
    conn.close()

    return [
        {"id": r[0], "name": r[1], "email": r[2], "role": r[3]}
        for r in rows
    ]


def get_employees_by_role(role: str) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, email, role FROM employees WHERE role = ?",
        (role,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {"id": r[0], "name": r[1], "email": r[2], "role": r[3]}
        for r in rows
    ]


def get_all_employees() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, email, role FROM employees")
    rows = cursor.fetchall()
    conn.close()

    return [
        {"id": r[0], "name": r[1], "email": r[2], "role": r[3]}
        for r in rows
    ]


# =========================
# ATTENDANCE WRITE
# =========================

def start_attendance(employee_id: int, date: str, start_time: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO attendance (employee_id, date, start_time)
        VALUES (?, ?, ?)
        """,
        (employee_id, date, start_time),
    )

    conn.commit()
    conn.close()


def end_attendance(employee_id: int, date: str, end_time: str):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE attendance
        SET end_time = ?
        WHERE employee_id = ? AND date = ?
        """,
        (end_time, employee_id, date),
    )

    conn.commit()
    conn.close()


# =========================
# ATTENDANCE READ (EMPLOYEE)
# =========================

def get_attendance_for_employee_on_date(employee_id: int, date: str) -> Optional[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT date, start_time, end_time
        FROM attendance
        WHERE employee_id = ? AND date = ?
        """,
        (employee_id, date),
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "date": row[0],
        "start_time": row[1],
        "end_time": row[2],
    }


def get_attendance_for_employee(employee_id: int) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT date, start_time, end_time
        FROM attendance
        WHERE employee_id = ?
        ORDER BY date
        """,
        (employee_id,),
    )

    rows = cursor.fetchall()
    conn.close()

    return [
        {"date": r[0], "start_time": r[1], "end_time": r[2]}
        for r in rows
    ]


# =========================
# ATTENDANCE READ (ORG LEVEL)
# =========================

def get_attendance_for_all_on_date(date: str) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT e.id, e.name, e.role, a.date, a.start_time, a.end_time
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        WHERE a.date = ?
        """,
        (date,),
    )

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "employee_id": r[0],
            "name": r[1],
            "role": r[2],
            "date": r[3],
            "start_time": r[4],
            "end_time": r[5],
        }
        for r in rows
    ]


def get_all_attendance() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT e.id, e.name, e.role, a.date, a.start_time, a.end_time
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        ORDER BY a.date
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "employee_id": r[0],
            "name": r[1],
            "role": r[2],
            "date": r[3],
            "start_time": r[4],
            "end_time": r[5],
        }
        for r in rows
    ]

# =========================
# ATTENDANCE SUMMARY
# =========================

def get_attendance_summary_for_date(date: str) -> Dict:
    """
    Returns summary for a given date:
    - total employees
    - employees with attendance
    - employees without attendance
    """

    conn = get_connection()
    cursor = conn.cursor()

    # Total employees
    cursor.execute("SELECT COUNT(*) FROM employees")
    total_employees = cursor.fetchone()[0]

    # Employees with attendance on given date
    cursor.execute(
        """
        SELECT COUNT(DISTINCT employee_id)
        FROM attendance
        WHERE date = ?
        """,
        (date,),
    )
    present_count = cursor.fetchone()[0]

    absent_count = total_employees - present_count

    conn.close()

    return {
        "date": date,
        "total_employees": total_employees,
        "present": present_count,
        "absent": absent_count,
    }
from tools.db_tool import (
    create_employee,
    get_employee_by_email,
    get_employee_by_id,
    get_all_employees,
    start_attendance,
    end_attendance,
    get_attendance_for_employee_on_date,
    get_attendance_summary_for_date,
)
from tools.time_tool import current_date


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_employee_crud():
    print_section("EMPLOYEE CRUD")

    emp_id = create_employee("Test User", "testuser@test.com", "QA Engineer")
    print("Created employee ID:", emp_id)

    emp = get_employee_by_id(emp_id)
    print("Fetched by ID:", emp)

    emp = get_employee_by_email("testuser@test.com")
    print("Fetched by Email:", emp)

    employees = get_all_employees()
    print("Total employees:", len(employees))


def test_attendance_flow():
    print_section("ATTENDANCE FLOW")

    employee = get_employee_by_email("testuser@test.com")
    if not employee:
        print("❌ Employee not found")
        return

    emp_id = employee["id"]
    today = current_date()

    print("Starting attendance...")
    start_attendance(emp_id, today, "10:00")

    record = get_attendance_for_employee_on_date(emp_id, today)
    print("After start:", record)

    print("Ending attendance...")
    end_attendance(emp_id, today, "18:00")

    record = get_attendance_for_employee_on_date(emp_id, today)
    print("After end:", record)


def test_attendance_summary():
    print_section("ATTENDANCE SUMMARY")

    today = current_date()
    summary = get_attendance_summary_for_date(today)

    print("Date:", summary["date"])
    print("Total Employees:", summary["total_employees"])
    print("Present:", summary["present"])
    print("Absent:", summary["absent"])


if __name__ == "__main__":
    test_employee_crud()
    test_attendance_flow()
    test_attendance_summary()
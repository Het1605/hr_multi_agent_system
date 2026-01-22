from graph.workflow import build_workflow
from graph.state import HRState


def run_test(message: str):
    print("\n" + "=" * 80)
    print("USER :", message)

    workflow = build_workflow()

    state = HRState(
        user_input=message,
        messages=[],
        intent=None,
        data={}
    )

    try:
        result = workflow.invoke(state)
    except Exception as e:
        print("❌ ERROR:", e)
        return

    print("INTENT:", result.get("intent"))

    messages = result.get("messages", [])
    if messages:
        print("BOT  :", messages[-1]["content"])
    else:
        print("BOT  : <no response>")


if __name__ == "__main__":
    test_cases = [

        # -------------------------
        # GREETING
        # -------------------------
        "hi",
        "who are you",

        # -------------------------
        # EMPLOYEE REGISTRATION
        # -------------------------
        "register smith smith@gmail.com node developer",
        "register harsh harsh@gmail.com java",
        "find employee smith",

        # -------------------------
        # ATTENDANCE START / END
        # -------------------------
        "smith start work at 10:00",
        "smith end work at 18:00",
        "harsh start work at 9:30",
        "harsh end work at 7 pm",

        # Combined start + end
        "yash work from 9 to 6",

        # Past date
        "smith start work yesterday at 10:00",
        "smith end work yesterday at 18:00",

        # Future date (should fail)
        "smith start work tomorrow at 10:00",

        # Update confirmation
        "smith start work at 11:00",
        "yes update it",

        # -------------------------
        # ATTENDANCE SUMMARY
        # -------------------------
        "attendance summary",
        "how many employees worked today",
        "who has not started work today",

        # -------------------------
        # REPORTS
        # -------------------------
        "daily report for smith today",
        "working hours of harsh today",
        "monthly report for smith january 2026",

        # -------------------------
        # HR POLICY
        # -------------------------
        "how many paid leaves do employees get",
        "what are office working hours",
        "is work from home allowed",
        "company rules",
    ]

    for msg in test_cases:
        run_test(msg)
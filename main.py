from graph.workflow import build_workflow
from graph.state import HRState


def main():
    app = build_workflow()

    print("🤖 HR Management System")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        initial_state: HRState = {
            "user_input": user_input,
            "intent": None,
            "employee_id": None,
            "data": {},
            "messages": []
        }

        try:
            result = app.invoke(initial_state)

            if result.get("messages"):
                print("Bot:", result["messages"][-1]["content"])
            else:
                print("Bot: I couldn't process your request.")

        except Exception as e:
            print("Bot: Something went wrong.")
            print("Error:", e)


if __name__ == "__main__":
    main()
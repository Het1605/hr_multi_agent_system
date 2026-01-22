from graph.workflow import build_workflow
from graph.state import HRState


def main():
    app = build_workflow()

    print("🤖 HR Management System")
    print("Type 'exit' to quit.\n")

    # Initialize state ONCE here (outside loop)
    state: HRState = {
        "user_input": "",
        "intent": None,
        "employee_id": None,
        "data": {},
        "messages": []
    }

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        # Update ONLY the input for the new turn
        state["user_input"] = user_input

        try:
            result = app.invoke(state)

            # Update persistent state with result
            # Crucial: Keep intent and data for continuity
            state["intent"] = result.get("intent")
            state["data"] = result.get("data", {})
            state["messages"] = result.get("messages", [])

            if result.get("messages"):
                print("Bot:", result["messages"][-1]["content"])
            else:
                print("Bot: I couldn't process your request.")

        except Exception as e:
            print("Bot: Something went wrong.")
            print("Error:", e)


if __name__ == "__main__":
    main()
from langgraph.graph import StateGraph, START, END

from graph.state import HRState
from graph.routing import route_by_intent

from agents.supervisor_agent import supervisor_agent
from agents.employee_agent import employee_agent
from agents.attendance_agent import attendance_agent
from agents.report_agent import report_agent
from agents.knowledge_agent import knowledge_agent


def build_workflow():
    graph = StateGraph(HRState)

    # -------------------------
    # Add nodes (agents)
    # -------------------------
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("employee_agent", employee_agent)
    graph.add_node("attendance_agent", attendance_agent)
    graph.add_node("report_agent", report_agent)
    graph.add_node("knowledge_agent", knowledge_agent)

    # -------------------------
    # Entry point
    # -------------------------
    graph.add_edge(START, "supervisor_agent")

    # -------------------------
    # Conditional routing
    # -------------------------
    graph.add_conditional_edges(
        "supervisor_agent",
        route_by_intent,
        {
            "employee_agent": "employee_agent",
            "attendance_agent": "attendance_agent",
            "report_agent": "report_agent",
            "knowledge_agent": "knowledge_agent",
        },
    )

    # -------------------------
    # End after each agent
    # -------------------------
    graph.add_edge("employee_agent", END)
    graph.add_edge("attendance_agent", END)
    graph.add_edge("report_agent", END)
    graph.add_edge("knowledge_agent", END)

    return graph.compile()
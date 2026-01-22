from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import HRState
from tools.vector_tool import similarity_search
from config.settings import LLM_MODEL, LLM_TEMPERATURE


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE
)


# -----------------------------
# STRONG POLICY PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an HR Policy Assistant for a company.

            You MUST answer using ONLY the provided policy context.
            Do NOT invent, assume, or hallucinate information.

            CRITICAL BEHAVIOR RULES:

            1. HIGH-LEVEL QUESTIONS (VERY IMPORTANT)
            If the user asks broad or umbrella questions such as:
            - "company rules"
            - "company policy"
            - "HR policies"
            - "overview of company policies"
            - "explain HR policies"
            - "rules"

            Then you MUST:
            - Summarize ALL relevant policies found in the context
            - Combine information across multiple policy sections
            - Provide a clear, structured summary using bullet points
            - NEVER reply with "not specified" if ANY policy information exists

            2. SPECIFIC QUESTIONS
            If the user asks about a specific topic (leave, working hours, WFH, overtime, etc.):
            - Answer ONLY that topic
            - If the topic is NOT mentioned in the context, reply exactly:
              "This information is not specified in the current company policies."

            3. SEMANTIC UNDERSTANDING
            Treat the following as equivalent:
            - "paid leaves" = "days off" = "leave days"
            - "office timings" = "working hours"
            - "company rules" = "company policies"

            4. STRICT CONTEXT BOUNDARY
            - NEVER add external HR knowledge
            - NEVER guess standard policies
            - NEVER mention internal files, vectors, embeddings, or context source

            5. RESPONSE STYLE
            - Be clear, professional, and concise
            - Prefer bullet points for summaries
            - Avoid unnecessary explanations

            6. FALLBACK RULE
            ONLY say "This information is not specified in the current company policies."
            IF AND ONLY IF:
            - The requested topic is completely absent from the provided context

            Remember:
            If ANY relevant policy exists -> summarize it.
            Do NOT default to "not specified" for broad questions.
            """
        ),
        ("human", "{input}")
    ]
)


# -----------------------------
# KNOWLEDGE AGENT
# -----------------------------
def knowledge_agent(state: HRState) -> Dict:
    user_input = state["user_input"]

    docs = similarity_search(user_input)

    # Rule 5: If no chunk passes the threshold, treat as "not found"
    if not docs:
        return {
            "messages": state.get("messages", []) + [
                {
                    "role": "assistant",
                    "content": "This information is not specified in the current company policies."
                }
            ]
        }

    context = "\n\n".join(docs)

    chain = prompt | llm
    response = chain.invoke(
        {
            "input": (
                f"Policy Context:\n{context}\n\n"
                f"User Question:\n{user_input}"
            )
        }
    )

    return {
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": response.content}
        ]
    }
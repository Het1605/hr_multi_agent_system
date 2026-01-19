from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from graph.state import HRState
from tools.vector_tool import similarity_search
from config.settings import LLM_MODEL, LLM_TEMPERATURE


llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an HR Knowledge Agent.

            Your responsibilities:
            - Answer HR policy and company rule questions
            - Use provided context only
            - If information is missing, say you don't know
            - Be clear and concise
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Context:
            {context}
            """
        )
    ]
)


def knowledge_agent(state: HRState) -> Dict:
    user_input = state["user_input"]

    try:
        documents: List[str] = similarity_search(user_input, k=3)
    except Exception:
        documents = []

    if not documents:
        response_text = (
            "I don’t have enough information in the HR knowledge base "
            "to answer this question right now."
        )
    else:
        context = "\n\n".join(documents)
        chain = prompt | llm
        response = chain.invoke(
            {
                "question": user_input,
                "context": context
            }
        )
        response_text = response.content

    return {
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": response_text}
        ]
    }
from typing import TypedDict, List, Any, Optional


class HRState(TypedDict):
    user_input: str
    intent: Optional[str]
    employee_id: Optional[int]
    data: Any
    messages: List[Any]
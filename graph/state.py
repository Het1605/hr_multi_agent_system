from typing import Dict, TypedDict, List, Any, Optional


class HRState(TypedDict):
    user_input: str
    intent: Optional[str]
    employee_id: Optional[int]
    data:Dict[str, Any]   # <-- entities live here
    messages: List[Any]
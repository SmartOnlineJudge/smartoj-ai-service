from typing import TypedDict, Literal
from datetime import datetime


ActionType = Literal["entry", "finish"]


class NodeCallLog(TypedDict):
    type: str
    action: ActionType
    name: str
    entried_time: str
    finished_time: str
    description: str


def create_node_call_log(name: str, description: str, action: ActionType) -> NodeCallLog:
    log = NodeCallLog(type="node_call_log", name=name, action=action, description=description)
    _time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if action == "entry":
        log["entried_time"] = _time
    elif action == "finish":
        log["finished_time"] = _time
    return log

from langgraph.graph import StateGraph

from core.state import SmartOJMessagesState
from .nodes.dispatcher import dispatch_next_node, DispatcherNode, path_map
from .nodes.question import QuestionNode
from .nodes.judge_template import JudgeTemplateNode
from .nodes.solving_framework import SolvingFrameworkNode
from .nodes.memory_time_limit import MemoryTimeLimitNode
from .nodes.test import TestNode
from .nodes.planner import PlannerNode


graph_builder = StateGraph(SmartOJMessagesState)

graph_builder.add_node("dispatcher", DispatcherNode())
graph_builder.add_node("question", QuestionNode())
graph_builder.add_node("judge_template", JudgeTemplateNode())
graph_builder.add_node("solving_framework", SolvingFrameworkNode())
graph_builder.add_node("memory_time_limit", MemoryTimeLimitNode())
graph_builder.add_node("test", TestNode())
graph_builder.add_node("planner", PlannerNode())

graph_builder.set_entry_point("question")
graph_builder.add_conditional_edges("dispatcher", dispatch_next_node, path_map)
graph_builder.add_edge("question", "dispatcher")
graph_builder.add_edge("judge_template", "dispatcher")
graph_builder.add_edge("solving_framework", "dispatcher")
graph_builder.add_edge("memory_time_limit", "dispatcher")
graph_builder.add_edge("test", "dispatcher")
graph_builder.add_edge("planner", "dispatcher")

graph = graph_builder.compile()

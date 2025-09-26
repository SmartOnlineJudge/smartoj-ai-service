from langgraph.graph import StateGraph

from core.state import SmartOJMessagesState
from .nodes.dispatcher import dispatch_next_node, DispatcherNode, target_node_mapping
from .nodes.question import QuestionNode
from .nodes.judge_template import JudgeTemplateNode
from .nodes.solving_framework import SolvingFrameworkNode
from .nodes.memory_time_limit import MemoryTimeLimitNode
from .nodes.test import TestNode


graph_builder = StateGraph(SmartOJMessagesState)

graph_builder.add_node("dispatcher", DispatcherNode())
graph_builder.add_node("question", QuestionNode())
graph_builder.add_node("judge_template", JudgeTemplateNode())
graph_builder.add_node("solving_framework", SolvingFrameworkNode())
graph_builder.add_node("memory_time_limit", MemoryTimeLimitNode())
graph_builder.add_node("test", TestNode())

graph_builder.set_entry_point("dispatcher")
graph_builder.add_conditional_edges("dispatcher", dispatch_next_node, target_node_mapping)
graph_builder.set_finish_point("question")
graph_builder.set_finish_point("judge_template")
graph_builder.set_finish_point("solving_framework")
graph_builder.set_finish_point("memory_time_limit")
graph_builder.set_finish_point("test")

graph = graph_builder.compile()

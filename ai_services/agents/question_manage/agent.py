from langgraph.graph import StateGraph
from langgraph.types import Checkpointer

from .nodes import (
    dispatcher_node, 
    dispatch_next_node,
    data_preheat_node,
    path_map,
    memory_time_limit_node,
    solving_framework_node,
    judge_template_node,
    test_node,
    planner_node
)
from .state import QuestionManageMessagesState


def build_question_manage_graph(checkpointer: Checkpointer = None, **kwargs):
    graph_builder = StateGraph(QuestionManageMessagesState)

    graph_builder.add_node("dispatcher", dispatcher_node)
    graph_builder.add_node("data_preheat", data_preheat_node)
    graph_builder.add_node("judge_template", judge_template_node)
    graph_builder.add_node("solving_framework", solving_framework_node)
    graph_builder.add_node("memory_time_limit", memory_time_limit_node)
    graph_builder.add_node("test", test_node)
    graph_builder.add_node("planner", planner_node)

    graph_builder.set_entry_point("data_preheat")
    graph_builder.add_conditional_edges("dispatcher", dispatch_next_node, path_map)
    graph_builder.add_edge("data_preheat", "dispatcher")
    graph_builder.add_edge("judge_template", "dispatcher")
    graph_builder.add_edge("solving_framework", "dispatcher")
    graph_builder.add_edge("memory_time_limit", "dispatcher")
    graph_builder.add_edge("test", "dispatcher")
    graph_builder.add_edge("planner", "dispatcher")

    return graph_builder.compile(checkpointer, **kwargs)

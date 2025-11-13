from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from nodes.check_plot_code import check_plot_code
from nodes.check_stats_code import check_stats_code
from nodes.choose_statistical_method_node import select_statistical_method_node
from nodes.conditional_edges import check_error_extract, check_error_method, max_iterations, decide_to_finish_plot
from nodes.plot_generator import plot_generator
from nodes.process_data_node import extract_sleep_data_node
from nodes.statistical_analysis_node import statistical_analysis_node
from type import State


# ==================== GRAPH CREATION ====================
def create_sleep_analysis_graph() -> CompiledStateGraph:
    """Crea la chain LangGraph con reflection per l'analisi"""

    workflow = StateGraph(State)

    # Aggiungi i nodi
    workflow.add_node("extract_data", extract_sleep_data_node)
    workflow.add_node("select_method", select_statistical_method_node)
    workflow.add_node("analyze", statistical_analysis_node)
    workflow.add_node("check_code", check_stats_code)
    workflow.add_node("plot_generator", plot_generator)
    workflow.add_node("check_plot", check_plot_code)

    # Entry point
    workflow.set_entry_point("extract_data")

    # ==================== EDGE CONNECTIONS ====================

    # extract_data -> [select_method OR end]
    workflow.add_conditional_edges(
        "extract_data",
        check_error_extract,
        {
            "select_method": "select_method",
            "end": END
        }
    )

    # select_method -> [analyze OR end]
    workflow.add_conditional_edges(
        "select_method",
        check_error_method,
        {
            "analyze": "analyze",
            "end": END
        }
    )

    # analyze -> check_code (always)
    workflow.add_edge("analyze", "check_code")

    # check_code -> [analyze (retry) OR plot_generator (success)]
    workflow.add_conditional_edges(
        "check_code",
        lambda state: "plot_generator" if state.get("error") == "no" else ("analyze" if state.get("iterations", 0) < max_iterations else "end"),
        {
            "analyze": "analyze",
            "plot_generator": "plot_generator",
            "end": END
        }
    )

    # plot_generator -> check_plot (always)
    workflow.add_edge("plot_generator", "check_plot")

    # check_plot -> [plot_generator (retry) OR end (success/max iterations)]
    workflow.add_conditional_edges(
        "check_plot",
        decide_to_finish_plot,  # Usa la nuova funzione
        {
            "plot_generator": "plot_generator",
            "end": END
        }
    )
    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)

# ==================== MAIN ====================
if __name__ == "__main__":
    app = create_sleep_analysis_graph()

    messages = []
    config = {
        "configurable": {
            "thread_id": "1"  # ID univoco per questa conversazione
        }
    }
    while True:
        query = input("Query").strip()

        if query.lower() == "end":
            break

        if not query:
            continue

        try:
            result = app.invoke({
                "query": query,
                "messages": messages,
                "iterations": 0
            }, config=config)
        except Exception as e:
            print(e)

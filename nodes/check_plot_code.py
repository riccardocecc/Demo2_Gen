
from type import State

from langchain_core.messages import HumanMessage, AIMessage

from utils.settings import python_repl

def check_plot_code(state: State) -> State:
    """
    Nodo: Verifica l'esecuzione del codice del grafico.
    """
    print("---CHECKING PLOT CODE---")

    messages = state.get("messages", [])
    plot_solution = state["plotly_figure"]
    raw_data = state.get("raw_data", {})
    imports = plot_solution.imports
    code = plot_solution.code

    try:
        python_repl.run(imports)
    except Exception as e:
        print(f"---PLOT IMPORT FAILED: {e}---")
        error_message = HumanMessage(content=f"Il codice di import del grafico ha fallito: {e}")
        return {
            **state,
            "messages": messages + [error_message],
            "error": "yes"
        }

    try:
        print("---CHECK PLOT CODE BLOCK---")
        python_repl.globals["result_graph"] = {}
        for key, records in raw_data.items():
            import pandas as pd
            df = pd.DataFrame(records)
            python_repl.globals[key] = df

        print(f"Executing plot code:\n{code}")

        python_repl.run(code)

        result_graph = python_repl.locals.get('result_graph')
        print("result_graph in locals", result_graph)
        if result_graph is None:
            result_graph = python_repl.globals.get('result_graph')
            print("result_graph in globals", result_graph)

        # Verifica che result_graph sia un dizionario valido
        if result_graph is None:
            print("---PLOT CODE FAILED: result_graph is None---")
            error_message = HumanMessage(
                content="Il codice del grafico non ha creato la variabile 'result_graph'. Assicurati di usare: result_graph = fig.to_dict()"
            )
            return {
                **state,
                "messages": messages + [error_message],
                "error": "yes"
            }

        # Verifica la struttura base del dizionario Plotly
        required_keys = ['data', 'layout']
        if not all(key in result_graph for key in required_keys):
            print(f"---PLOT CODE FAILED: Missing keys. Has {list(result_graph.keys())}---")
            error_message = HumanMessage(
                content=f"Il grafico non ha la struttura Plotly corretta. Dovrebbe contenere: {required_keys}. Contiene: {list(result_graph.keys())}"
            )
            return {
                **state,
                "messages": messages + [error_message],
                "error": "yes"
            }

    except Exception as e:
        print("---PLOT CODE CHECK: FAILED---")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}")

        import traceback
        print("FULL TRACEBACK:")
        traceback.print_exc()

        error_message = HumanMessage(content=f"Il codice del grafico ha fallito: {e}")
        return {
            **state,
            "messages": messages + [error_message],
            "error": "yes"
        }

    # Salva l'HTML solo in caso di successo
    print("---PLOT CODE CHECK: SUCCESS---")

    try:
        import plotly.graph_objects as go
        from pathlib import Path

        fig = go.Figure(result_graph)
        html = fig.to_html(full_html=True, include_plotlyjs='cdn')

        project_root = Path(__file__).parent
        out_path = project_root / "plot_debug.html"
        out_path.write_text(html, encoding='utf-8')

        print(f"Saved debug HTML to: {out_path}")
    except Exception as e:
        print(f"Warning: Could not save HTML file: {e}")

    success_message = AIMessage(content="Grafico generato con successo!")

    return {
        **state,
        "messages": messages + [success_message],
        "error": "no",  # ← Questo è CRITICO
        "plotly_figure_dict": result_graph
    }


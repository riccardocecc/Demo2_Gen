import numpy as np
from type import State

from langchain_core.messages import HumanMessage, AIMessage

from utils.settings import python_repl


def check_stats_code(state: State) -> State:
    """
    Nodo 3: Verifica l'esecuzione del codice generato.
    """
    print("--Checking code solution--")

    messages = state.get("messages", [])
    code_solution = state["generation"]
    iterations = state.get("iterations", 0)
    raw_data = state.get("raw_data", {})
    imports = code_solution.imports
    code = code_solution.code

    try:
        print("---CHECK IMPORTS---")
        python_repl.run(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = HumanMessage(content=f"Your solution failed the import test: {e}")
        return {
            **state,
            "messages": messages + [error_message],
            "error": "yes"
        }

    try:
        print("---CHECK CODE BLOCK---")
        python_repl.globals["result"]={}
        for key, records in raw_data.items():
            import pandas as pd
            df = pd.DataFrame(records)
            python_repl.globals[key] = df
            print(f"Loaded {key} into globals")

        print(f"Executing code:\n{code}")
        python_repl.run(code)
        if python_repl.locals["result"]:
            print("Result in locals", python_repl.locals["result"])

        result = python_repl.globals.get('result')
        print(f"Result in globals: {result}")

        # Check if result exists and is a dict
        if result is None:
            print("---CODE BLOCK CHECK: FAILED---")
            error_message = HumanMessage(
                content="Your solution failed: 'result' variable was not created or is None. Make sure to store the output in a variable named 'result'."
            )
            return {
                **state,
                "messages": messages + [error_message],
                "error": "yes"
            }

        # Only check for NaN in numeric values, not None values which are intentional
        def contains_invalid_nan(obj):
            """Check for NaN values only in float/numeric types, not None"""
            if obj is None:
                return False  # None is valid
            if isinstance(obj, dict):
                return any(contains_invalid_nan(v) for v in obj.values())
            if isinstance(obj, (list, tuple, set)):
                return any(contains_invalid_nan(v) for v in obj)
            if isinstance(obj, str):
                return False  # Strings are valid
            if isinstance(obj, np.generic):
                try:
                    return np.isnan(obj)
                except (TypeError, ValueError):
                    return False
            if isinstance(obj, float):
                return np.isnan(obj)
            return False

        if contains_invalid_nan(result):
            print("---CODE BLOCK CHECK: FAILED---")
            print("ERROR TYPE: ValueError")
            print("ERROR MESSAGE: Result contains NaN values in numeric fields")

            error_message = HumanMessage(
                content=f"Your solution failed: result contains NaN values in numeric fields. Either handle NaN values properly or set them to None explicitly. Result: {result}"
            )

            return {
                **state,
                "messages": messages + [error_message],
                "error": "yes"
            }

    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}")

        import traceback
        print("FULL TRACEBACK:")
        traceback.print_exc()

        error_message = HumanMessage(content=f"Your solution failed the code execution test: {e}")

        return {
            **state,
            "messages": messages + [error_message],
            "error": "yes"
        }

    print("---CODE CHECK: SUCCESS---")
    success_message = AIMessage(content=f"Code executed successfully. Result: {result}")
    state["code_response"] = result
    return {
        **state,
        "iterations": 0,
        "messages": messages + [success_message],
        "error": "no"
    }

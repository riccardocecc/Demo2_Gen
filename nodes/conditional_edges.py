from typing import Literal

from type import State
max_iterations = 3


# ==================== CONDITIONAL EDGES ====================
def check_error_extract(state: State) -> Literal["select_method", "end"]:
    """Controlla se ci sono errori dopo l'estrazione dati"""
    return "end" if state.get("error") else "select_method"

def check_error_method(state: State) -> Literal["analyze", "end"]:
    """Controlla se ci sono errori dopo la selezione del metodo"""
    return "end" if state.get("error") else "analyze"

def decide_to_finish(state: State) -> Literal["analyze", "end"]:
    """
    Decide se continuare con un'altra iterazione o terminare.
    - Se error == "no": termina con successo
    - Se iterations >= max_iterations: termina per troppi tentativi
    - Altrimenti: riprova con analyze
    """
    error = state.get("error", "")
    iterations = state.get("iterations", 0)

    if error == "no":
        print("---DECISION: SUCCESS - FINISH---")
        return "end"
    elif iterations >= max_iterations:
        print(f"---DECISION: MAX ITERATIONS ({max_iterations}) REACHED - FINISH---")
        return "end"
    else:
        print(f"---DECISION: RETRY ANALYZE (attempt {iterations + 1}/{max_iterations})---")
        return "analyze"

def decide_to_finish_plot(state: State) -> Literal["plot_generator", "end"]:
    """
    Decide se continuare con il plot o terminare.
    """
    error = state.get("error", "")
    plot_attempts = state.get("plot_attempts", 0)

    print(f"\n---PLOT DECISION CHECK---")
    print(f"Error status: {error}")
    print(f"Plot attempts: {plot_attempts}")

    if error == "no":
        print("---PLOT DECISION: SUCCESS - FINISH---")
        return "end"
    elif plot_attempts >= 3:  # Aumentato a 3 tentativi
        print(f"---PLOT DECISION: MAX ATTEMPTS ({plot_attempts}) REACHED - FINISH---")
        return "end"
    else:
        print(f"---PLOT DECISION: RETRY PLOT (attempt {plot_attempts + 1}/3)---")
        return "plot_generator"

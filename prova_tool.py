from typing import TypedDict, Literal, List, Annotated, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from langchain_experimental.utilities.python import PythonREPL

from code_chain import create_code_chain, max_iterations
from datacleaner import DataCleaner
from domain_configs import SLEEP_CONFIG, KITCHEN_CONFIG
from registry.DomainRegistry import domain_registry
from settings import client_oll as llm_code, oll_tool
from tool import get_sleep_data, get_kitchen_data

python_repl = PythonREPL()


class StatisticalMethodSelection(BaseModel):
    """Schema per la selezione del metodo statistico"""
    analysis_goal: str = Field(
        description="Riformula brevemente l'obiettivo della query SENZA aggiungere dettagli non richiesti"
    )
    analysis_type: str = Field(
        description="Tipo di analisi (es: correlation, proportion, descriptive, trend, comparison, etc.)"
    )
    variables: List[str] = Field(
        description="Lista delle colonne da analizzare (usa i nomi esatti delle colonne)"
    )
    statistical_methods: List[str] = Field(
        description="Metodi statistici da applicare (es: Pearson correlation, t-test, mean, sum, etc.)"
    )
    calculations_needed: Dict[str, str] = Field(
        description="Dizionario con nome metrica e descrizione del calcolo"
    )
    expected_outputs: List[str] = Field(
        description="Lista degli output attesi dall'analisi"
    )
class State(TypedDict):
    """State che mantiene lo stato della conversazione attraverso i nodi"""
    query: str
    subject_id: int
    period: str
    raw_data: dict[str, any]
    domains_detected: list[str]
    statistical_method: dict
    error: str
    messages: Annotated[List[BaseMessage], add_messages]
    generation: str
    iterations: int
    plotly_figure: dict
    plot_attempts: int
    plot_errors: list
    final_response: str


max_iterations = 3


# ==================== UTILITY FUNCTIONS ====================
def process_sleep_data(result: dict, verbose: bool = True) -> pd.DataFrame:
    """Pulisce i dati del sonno usando DataCleaner"""
    cleaner = DataCleaner(SLEEP_CONFIG, verbose=verbose)
    return cleaner.clean(result['records'])


def process_kitchen_data(result: dict, verbose: bool = True) -> pd.DataFrame:
    cleaner = DataCleaner(KITCHEN_CONFIG, verbose=verbose)
    return cleaner.clean(result['records'])


# ==================== NODE 1: DATA EXTRACTION ====================
def extract_sleep_data_node(state: State) -> State:
    """
    Nodo 1: Estrae i dati dal CSV usando la query in linguaggio naturale.
    """
    print("\n[NODE 1] Estrazione dati dal CSV...")

    query = state["query"]
    llm_with_tools = oll_tool.bind_tools([get_sleep_data, get_kitchen_data])

    extraction_prompt = f"""Recupera dati per questa query: "{query}"

    Tool disponibili:
    - get_sleep_data: per dati sul sonno
    - get_kitchen_data: per dati sulla cucina

    Estrai:
    - subject_id (default: 1)
    - period (default: 'last_30_days')

    Chiama ENTRAMBI i tool se la query riguarda entrambi i domini."""

    result_tool: dict[str, any] = {}
    domains_detected: list[str] = []

    try:
        # Crea messaggio usando HumanMessage
        messages = [HumanMessage(content=extraction_prompt)]
        response = llm_with_tools.invoke(messages)

        if isinstance(response, AIMessage) and response.tool_calls:
            print("tool calls:", response.tool_calls)

            for tool in response.tool_calls:
                subject_id = tool['args'].get('subject_id', 1)
                period = tool['args'].get('period', 'last_30_days')

                if tool['name'] == 'get_sleep_data':
                    result = get_sleep_data.invoke({
                        'subject_id': subject_id,
                        'period': period
                    })
                    # Pulisci i dati
                    df_clean = process_sleep_data(result, verbose=False)

                    # Converti in dizionario SENZA valori None/NaN
                    cleaned_records = df_clean.replace({np.nan: None}).to_dict('records')
                    # Rimuovi le chiavi con valore None
                    cleaned_records = [
                        {k: v for k, v in record.items() if v is not None}
                        for record in cleaned_records
                    ]

                    result_tool[tool['name']] = cleaned_records
                    domains_detected.append('sleep')

                elif tool['name'] == 'get_kitchen_data':
                    result = get_kitchen_data.invoke({
                        'subject_id': subject_id,
                        'period': period
                    })
                    # Pulisci i dati
                    df_clean = process_kitchen_data(result, verbose=False)

                    # Converti in dizionario SENZA valori None/NaN
                    cleaned_records = df_clean.replace({np.nan: None}).to_dict('records')
                    # Rimuovi le chiavi con valore None
                    cleaned_records = [
                        {k: v for k, v in record.items() if v is not None}
                        for record in cleaned_records
                    ]

                    result_tool[tool['name']] = cleaned_records
                    domains_detected.append('kitchen')

        print(f"✓ Estratti parametri - soggetto ID: {subject_id}, period: {period}")
        print(result_tool)

        return {
            **state,
            "subject_id": subject_id,
            "period": period,
            "raw_data": result_tool,
            "domains_detected": domains_detected,
            "messages": [response]
        }

    except Exception as e:
        error_msg = f"Errore nell'estrazione dati: {str(e)}"
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

        return {
            **state,
            "error": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }


# ==================== NODE 1.5: STATISTICAL METHOD SELECTION ====================
def select_statistical_method_node(state: State) -> State:
    """
    Nodo 1.5: Analizza la query e determina quale analisi statistica è più appropriata.
    """
    print("\n[NODE 1.5] Selezione metodo statistico...")

    if state.get("error"):
        return state

    domains_detected = state.get("domains_detected", [])
    available_columns = domain_registry.get_available_columns_for_domains(domains_detected)
    query = state["query"]

    print("Numero di record disponibili:", len(available_columns))

    system_prompt = """Sei un esperto statistico specializzato nell'analisi dei dati.

COMPITO: 
Analizza la query e determina l'approccio statistico più appropriato per rispondere alla domanda.
Hai TOTALE LIBERTÀ nella scelta del metodo e delle tecniche statistiche.

CONSIDERAZIONI:
1. Quale domanda sta facendo l'utente?
2. Quali variabili sono coinvolte?
3. Qual è il tipo di analisi più appropriato? (descrittiva, correlazione, confronto, trend, proporzioni, ecc.)
4. Quali metriche statistiche sono necessarie?

ISTRUZIONI:
1. Identifica l'obiettivo principale della query
2. Determina quale approccio statistico risponde meglio alla domanda
3. Identifica le variabili specifiche da analizzare (usa i nomi esatti delle colonne disponibili)
4. Specifica il metodo statistico da applicare più opportuno
5. Indica quali metriche e output calcolare

IMPORTANTE: Usa SOLO i nomi delle colonne presenti nei dati disponibili."""

    user_prompt = f"""Query dell'utente: "{query}"

Dati disponibili (campi per ogni record):
{available_columns}

Analizza la query e determina il metodo statistico più appropriato."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    structured_llm = llm_code.with_structured_output(StatisticalMethodSelection)

    try:
        method_selection = structured_llm.invoke(messages)

        method_dict = method_selection.model_dump()

        print(method_dict)
        print(f"✓ Obiettivo analisi: {method_dict['analysis_goal']}")
        print(f"  Tipo analisi: {method_dict['analysis_type']}")
        print(f"  Variabili: {method_dict['variables']}")

        return {
            **state,
            "statistical_method": method_dict,
            "messages": state.get("messages", []) + messages
        }

    except Exception as e:
        print(f"Errore nella selezione del metodo statistico: {e}")
        import traceback
        traceback.print_exc()
        return {
            **state,
            "error": "yes",
            "error_message": f"Failed to select statistical method: {e}"
        }

# ==================== NODE 2: STATISTICAL ANALYSIS WITH REFLECTION ====================
def statistical_analysis_node(state: State) -> State:
    """
    Nodo 2: Genera codice Python per analisi statistica usando Pydantic structured output.
    """
    print("--Generating code solution--")

    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    error = state.get("error", "")
    domains_detected = state.get("domains_detected", [])
    available_columns_by_df = domain_registry.get_available_columns_for_domains(domains_detected)

    query = state["query"]
    method_selection = state.get("statistical_method", {})
    raw_data = state.get("raw_data", {})

    available_dataframes = list(raw_data.keys())
    subject = state["subject_id"]

    if error == "yes":
        messages = messages + [
            HumanMessage(
                content="Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:")
        ]

    # Formatta il context come testo leggibile invece di dizionari Python grezzi
    context = f"""Query: "{query}"

Analysis Details:
- Analysis Goal: {method_selection.get('analysis_goal', 'N/A')}
- Analysis Type: {method_selection.get('analysis_type', 'N/A')}
- Variables: {', '.join(method_selection.get('variables', []))}
- Statistical Methods: {', '.join(method_selection.get('statistical_methods', []))}
- Calculations Needed: {', '.join(f"{k}: {v}" for k, v in method_selection.get('calculations_needed', {}).items())}
- Expected Outputs: {', '.join(method_selection.get('expected_outputs', []))}
- Visualization Type: {method_selection.get('visualization_type', 'N/A')}

Available columns:
{chr(10).join(f"- {df_name}: {', '.join(f'{col} ({dtype})' for col, dtype in columns.items())}" for df_name, columns in available_columns_by_df.items())}

Subject to analyze: {subject}

IMPORTANT:
A pandas DataFrame named is already available in the environment:
1. Use ONLY these DataFrames: {', '.join(available_dataframes)}  
2. Access dataframes using their EXACT names: {', '.join(available_dataframes)}
3. Use ONLY these variables in your analysis: {', '.join(method_selection.get('variables', []))}
4. DO NOT filter data by dates!! - the data is already filtered for the correct time period
5. Work with ALL rows in the DataFrames provided
6. NO GRAPH 
"""

    code_gen_chain = create_code_chain(context)

    code_solution = code_gen_chain.invoke({"messages": messages})

    assistant_message = AIMessage(
        content=f"{code_solution.description}\n\nImports:\n{code_solution.imports}\n\nCode:\n{code_solution.code}"
    )

    iterations = iterations + 1

    return {
        **state,
        "generation": code_solution,
        "messages": messages + [assistant_message],
        "iterations": iterations
    }

def code_check(state: State) -> State:
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

        for key, records in raw_data.items():
            import pandas as pd
            df = pd.DataFrame(records)
            python_repl.globals[key] = df
            print(f"Loaded {key} into globals")

        print(f"Executing code:\n{code}")
        python_repl.run(code)

        result = python_repl.locals.get('result')
        print(f"Result: {result}")

    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}")

        # QUESTO È CRITICO - stampa il traceback completo
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

    return {
        **state,
        "messages": messages + [success_message],
        "error": "no"
    }


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


# ==================== GRAPH CREATION ====================
def create_sleep_analysis_graph() -> CompiledStateGraph:
    """Crea la chain LangGraph con reflection per l'analisi"""

    workflow = StateGraph(State)

    # Aggiungi i nodi
    workflow.add_node("extract_data", extract_sleep_data_node)
    workflow.add_node("select_method", select_statistical_method_node)
    workflow.add_node("analyze", statistical_analysis_node)
    workflow.add_node("check_code", code_check)

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

    # check_code -> [analyze (retry) OR end (success/max iterations)]
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "analyze": "analyze",  # Riprova
            "end": END  # Termina
        }
    )

    return workflow.compile()


# ==================== MAIN ====================
if __name__ == "__main__":
    app = create_sleep_analysis_graph()

    result = app.invoke({
        "query": "C'è qualche correlazione tra il numero di risvegli e l'utilizzo della cucina per il soggetto 2 negli ultimi 10 giorni?",
        "messages": [],
        "iterations": 0
    })
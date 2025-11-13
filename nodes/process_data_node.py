# ==================== UTILITY FUNCTIONS ====================
from utils.datacleaner import DataCleaner
from utils.domain_configs import SLEEP_CONFIG, KITCHEN_CONFIG
import pandas as pd
from type import  State
from utils.settings import client_oll as llm_code
from tool import get_sleep_data, get_kitchen_data
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np

def process_sleep_data(result: dict, verbose: bool = True) -> pd.DataFrame:
    """Pulisce i dati del sonno usando DataCleaner"""
    print(result)
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
    llm_with_tools = llm_code.bind_tools([get_sleep_data, get_kitchen_data])

    extraction_prompt = f"""Recupera dati per questa query: "{query}" non considerare il resto, estrai solamente le informaizoni per invocare:
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

                    cleaned_records = df_clean.replace({np.nan: None}).to_dict('records')

                    cleaned_records = [
                        {k: v for k, v in record.items() if v is not None}
                        for record in cleaned_records
                    ]

                    result_tool[tool['name']] = cleaned_records
                    domains_detected.append('kitchen')

        print(f"✓ Estratti parametri - soggetto ID: {subject_id}, period: {period}")

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


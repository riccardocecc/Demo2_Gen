from type import State, StatisticalMethodSelection
from registry.DomainRegistry import domain_registry
from langchain_core.messages import HumanMessage, SystemMessage
from utils.settings import client_oll as llm_code


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
6. Se necessario, descrivi brevemente che tipo di visualizzazione plotly è necessaria, NO GRAPH altrimenti

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


from registry.DomainRegistry import domain_registry
from type import State

from langchain_core.messages import HumanMessage, AIMessage

from utils.code_chain import create_code_chain
def plot_generator(state: State) -> State:
    """
    Nodo: Genera codice per creare un grafico Plotly basato sui risultati dell'analisi statistica.
    """
    print("\n[NODE PLOT] Generazione grafico Plotly...")

    code_result = state.get("code_response", {})
    statistical_method = state["statistical_method"]
    query_user = state.get("query")
    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    error = state.get("error", "")
    available_dataframes = list(state["raw_data"].keys())


    available_columns_by_df = domain_registry.get_available_columns_for_domains(state["domains_detected"])

    if error == "yes":
        messages = messages + [
            HumanMessage(
                content="Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:")
        ]


    analysis_type = statistical_method.get('analysis_type', 'unknown')
    variables = statistical_method.get('variables', [])



    #INUTILMENTE COMPLICATO; ELIMINABILE
    columns_info = chr(10).join(
        f"  • {df_name}: {', '.join(f'{col} ({dtype})' for col, dtype in columns.items())}"
        for df_name, columns in available_columns_by_df.items()
    )
    visual_type = state.get("statistical_method", {}).get("visualization_type", "unknown")
    context = f"""OBIETTIVO: Crea un grafico Plotly che visualizzi i risultati dell'analisi statistica.

DOMANDA UTENTE: "{query_user}"

DATI ANALISI COMPLETATA:
- Tipo di analisi: {analysis_type}
- Variabili analizzate: {', '.join(variables)}
- Risultati statistici ottenuti:
{code_result}

DATAFRAME DISPONIBILI:
{', '.join(available_dataframes)}

IMPORTANTE: I DataFrames contengono i seguenti campi:
{columns_info}

REQUISITI DEL GRAFICO:
1. {visual_type}


2. Il grafico DEVE:
   - Visualizzare le variabili: {', '.join(variables)}
   - Includere titoli descrittivi e labels degli assi
   - DataFrame già presente in globals con nome: {available_dataframes[0]}
   - Se l'analisi ha prodotto coefficienti/statistiche, mostrarli come annotazioni o nel titolo

3. STRUTTURA DEL CODICE RICHIESTA:
   Step 1: Usa il DataFrame già presente globale esatto: {available_dataframes[0]}
   Step 2: Crea il grafico con plotly.express o plotly.graph_objects
   Step 3: Personalizza titolo, labels, colori
   Step 4: Se correlation, aggiungi trendline="ols"
   Step 5: IMPORTANTE - Converti in dizionario: result_graph = fig.to_dict()

4. BEST PRACTICES:
   - Usa colori professionali
   - Aggiungi hover_data per informazioni aggiuntive
   - Includi i risultati numerici importanti nel titolo del grafico
   - Per correlation: mostra il coefficiente di correlazione e p-value nel titolo

5. ESEMPIO SPECIFICO PER QUESTO CASO:
   Dato che l'analisi è di tipo correlation, crea uno scatter plot con:
   - Asse X: prima variabile della lista
   - Asse Y: seconda variabile della lista  
   - Aggiungi linea di tendenza con trendline="ols"
   - Titolo che include i valori statistici calcolati
   - Labels in italiano per gli assi

NOTA CRITICA: 
- La variabile globale result_graph DEVE contenere il grafico!!!!
- NON lasciare result_graph vuoto o None
- IMPORTANTE - Converti in dizionario: result_graph = fig.to_dict()
"""

    plot_chain = create_code_chain(
        context=context,
        result_var=" in globals variable result_graph",
        result_format="using fig.to_dict() to convert the Plotly figure to a dictionary"
    )

    plot_generated = plot_chain.invoke({"messages": messages})

    assistant_message = AIMessage(
        content=f"{plot_generated.description}\n\nImports:\n{plot_generated.imports}\n\nCode:\n{plot_generated.code}"
    )



    plot_attempts = state.get("plot_attempts", 0) + 1

    return {
        **state,
        "plotly_figure": plot_generated,
        "messages": messages + [assistant_message],
        "plot_attempts": plot_attempts,  # Usa plot_attempts invece di iterations
    }

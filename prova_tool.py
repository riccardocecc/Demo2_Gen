from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from langchain_experimental.tools import PythonREPLTool

from datacleaner import DataCleaner
from domain_configs import SLEEP_CONFIG, KITCHEN_CONFIG
from registry.DomainRegistry import domain_registry
from settings import client_oll as llm_code, oll_tool
from tool import get_sleep_data, get_kitchen_data

python_repl = PythonREPLTool()


# ==================== STATE DEFINITION ====================
class StatisticalAnalysisCode(BaseModel):
    """Schema per il codice di analisi statistica"""
    explanation: str = Field(description="Brief explanation of statistical approach (1-2 sentences)")
    imports: str = Field(description="Additional Python imports if needed (empty string if none)")
    code: str = Field(description="Executable Python code that populates 'results' dict and ends with print(json.dumps(results, default=str))")


class PlotlyGraphCode(BaseModel):
    imports: str = Field(
        description="Additional Python imports if needed. Use empty string '' if no extra imports needed. Example: '' or 'from datetime import datetime'",
        default=""
    )
    code: str = Field(
        description="""EXECUTABLE Python code (not a description!) that:
1. Filters/processes the dataframe
2. Creates a Plotly figure and assigns it to variable 'fig'
3. Ends with: print(fig.to_json())

EXAMPLE (this is ACTUAL code, not a description):
df_filtered = get_sleep_data[get_sleep_data['subject_id'] == 2].tail(10)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_filtered['data'], y=df_filtered['wakeup_count'], mode='lines+markers'))
fig.update_layout(title='Risvegli')
print(fig.to_json())

DO NOT write: 'The code to generate...' - Write ACTUAL executable Python code!"""
    )

class SleepAnalysisState(TypedDict):
    """State che mantiene lo stato della conversazione attraverso i nodi"""
    query: str
    subject_id: int
    period: str
    raw_data: dict[str, any]
    domains_detected: list[str]
    statistical_method: dict
    analysis_code: str
    analysis_results: dict
    analysis_imports: str
    analysis_errors: list
    analysis_attempts: int
    plotly_figure: dict
    plot_attempts: int
    plot_errors: list
    error: str
    final_response: str


# ==================== UTILITY FUNCTIONS ====================
def process_sleep_data(result: dict, verbose: bool = True) -> pd.DataFrame:
    """Pulisce i dati del sonno usando DataCleaner"""
    cleaner = DataCleaner(SLEEP_CONFIG, verbose=verbose)
    return cleaner.clean(result['records'])

def process_kitchen_data(result: dict, verbose: bool = True) -> pd.DataFrame:
    cleaner = DataCleaner(KITCHEN_CONFIG, verbose=verbose)
    return cleaner.clean(result['records'])

# ==================== NODE 1: DATA EXTRACTION ====================
def extract_sleep_data_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 1: Estrae i dati dal CSV usando la query in linguaggio naturale.
    """
    print("\n[NODE 1] Estrazione dati dal CSV...")

    query = state["query"]
    llm_with_tools = llm.bind_tools([get_sleep_data, get_kitchen_data])

    extraction_prompt = f"""
    /no thinking
    Sei un assistente sanitario che deve recuperare dati, hai a disposizione due tool:
    1) get_sleep_data: chiama questo tool se la query si riferisce al sonno
    2) get_kitchen_data: chiama questo tool se la query si riferisce alla cucina
    IMPORTANTE!: se la query comprende entrambi i domini chiama ENTRAMBI i tool

    Devi SOLO ESTRARRE I PARAMETRI NON CONSIDERARE TUTTO IL RESTO.

    Query dell'utente: "{query}"

    Analizza la query ed estrai:
    1. subject_id: ID del soggetto (numero intero). Se non specificato, usa 1 come default.
    2. period: periodo da analizzare in formato 'last_N_days' (es: 'last_30_days') oppure 'YYYY-MM-DD,YYYY-MM-DD'
       Se non specificato, usa 'last_30_days' come default.
    """
    result_tool: dict[str, any] = {}
    domains_detected: list[str] = []

    try:
        response = llm_with_tools.invoke(query, reasoning=False, think=False)
        if isinstance(response, AIMessage) and response.tool_calls:
            print("toolllll",response.tool_calls)

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
                }, reasoning=False)
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
        state["subject_id"] = subject_id
        state["period"] = period
        state["raw_data"] = result_tool
        state["domains_detected"] = domains_detected

    except Exception as e:
        state["error"] = f"Errore nell'estrazione dati: {str(e)}"
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    return state
# ==================== NODE 1.5: STATISTICAL METHOD SELECTION ====================
def select_statistical_method_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 1.5: Analizza la query e determina quale analisi statistica è più appropriata.
    """
    print("\n[NODE 1.5] Selezione metodo statistico...")

    if state.get("error"):
        return state
    domains_detected = state.get("domains_detected", [])
    available_columns = domain_registry.get_available_columns_for_domains(domains_detected)
    available_columns_by_df = domain_registry.get_available_columns_for_domains(domains_detected)
    query = state["query"]

    print("Numero di record disponibili:", len(available_columns))

    selection_prompt = f"""
    /no thinking
    Sei un esperto statistico specializzato nell'analisi dei dati.

    Query dell'utente: "{query}"

    Dati disponibili (campi per ogni record):
        {available_columns}

    COMPITO: 
    Analizza la query e determina l'approccio statistico più appropriato per rispondere alla domanda.
    Hai TOTALE LIBERTÀ nella scelta del metodo e delle tecniche statistiche.

    CONSIDERAZIONI:
    1. Quale domanda sta facendo l'utente?
    2. Quali variabili sono coinvolte?
    3. Qual è il tipo di analisi più appropriato? (descrittiva, correlazione, confronto, trend, proporzioni, ecc.)
    4. Quali metriche statistiche sono necessarie?
    5. Quale tipo di visualizzazione sarebbe più efficace?

    ISTRUZIONI:
    1. Identifica l'obiettivo principale della query
    2. Determina quale approccio statistico risponde meglio alla domanda
    3. Identifica le variabili specifiche da analizzare (usa i nomi esatti delle colonne)
    4. Specifica il metodo statistico da applicare più opportuno
    5. Indica quali metriche e output calcolare
    6. Suggerisci il tipo di visualizzazione più appropriato

    Rispondi in formato JSON con questa struttura:
    {{
        "analysis_goal": "riformula brevemente l'obiettivo della query SENZA aggiungere dettagli non richiesti",
        "analysis_type": "tipo di analisi (es: correlation, proportion, descriptive, trend, comparison, etc.)",
        "variables": ["lista", "delle", "colonne", "da", "analizzare"],
        "statistical_methods": ["metodo"],
        "calculations_needed": {{
            "metric1": "descrizione calcolo",
            "metric2": "descrizione calcolo"
        }},
        "expected_outputs": ["output1", "output2"],
        "visualization_type": "tipo di grafico suggerito",
    }}

    Rispondi SOLO con il JSON, senza markdown o spiegazioni aggiuntive.
    """

    try:
        response = llm.invoke([HumanMessage(content=selection_prompt)],reasoning=False)

        response_text = response.content.strip()
        response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)

        # Rimuovi i blocchi di codice markdown
        response_text = re.sub(r'```json\n?', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)

        method_selection = json.loads(response_text)
        print(method_selection)
        state["statistical_method"] = method_selection

        print(f"✓ Obiettivo analisi: {method_selection['analysis_goal']}")
        print(f"  Tipo analisi: {method_selection['analysis_type']}")
        print(f"  Variabili: {method_selection['variables']}")

    except json.JSONDecodeError:
        print(f"⚠️ Errore nel parsing JSON, uso fallback...")
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["correlazione", "relazione"]):
            analysis_type = "correlation"
            viz_type = "scatter"
        elif any(kw in query_lower for kw in ["proporzione", "percentuale"]):
            analysis_type = "proportion"
            viz_type = "pie"
        elif any(kw in query_lower for kw in ["trend", "andamento"]):
            analysis_type = "trend"
            viz_type = "line"
        else:
            analysis_type = "descriptive"
            viz_type = "bar"

        method_selection = {
            "analysis_goal": "Analisi automatica",
            "analysis_type": analysis_type,
            "variables": [],
            "statistical_methods": ["auto-detected"],
            "calculations_needed": {},
            "expected_outputs": [],
            "visualization_type": viz_type,
            "considerations": ""
        }
        state["statistical_method"] = method_selection

    except Exception as e:
        state["error"] = f"Errore selezione metodo: {str(e)}"

    return state


# ==================== NODE 2: STATISTICAL ANALYSIS WITH REFLECTION ====================
def statistical_analysis_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 2: Genera codice Python per analisi statistica usando Pydantic structured output.
    """
    print(f"\n[NODE 2] Analisi statistica - Tentativo {state['analysis_attempts'] + 1}...")

    if state.get("error"):
        return state
    domains_detected = state.get("domains_detected", [])
    available_columns_by_df = domain_registry.get_available_columns_for_domains(domains_detected)

    query = state["query"]
    method_selection = state.get("statistical_method", {})
    raw_data = state.get("raw_data", {})
    state["analysis_attempts"] += 1


    available_dataframes = list(raw_data.keys())
    subject = state["subject_id"]

    base_prompt = f"""
        /no thinking
        Generate Python code for statistical analysis using statsmodels.
    
        Query: "{query}"
    
        Analysis Details:
        {method_selection}
    
        Available columns:
        {available_columns_by_df}
        
        Subject to analyze
        {subject}
    
        Libraries ALREADY imported (do NOT re-import these):
        - pandas as pd
        - numpy as np
        - statsmodels.api as sm
        - statsmodels.formula.api as smf
        - json
    
        Variables ALREADY available in execution context:
        {', '.join([f'- {df_name}: pandas DataFrame' for df_name in available_dataframes])}
        - results: empty dictionary {{}} to populate with your analysis results
    
        CRITICAL Requirements:
        1. Use ONLY statsmodels for statistical calculations (NO scipy.stats, NO sklearn)
        2. Use ONLY these DataFrames: {', '.join(available_dataframes)}  
        3. Access dataframes using their EXACT names: {', '.join(available_dataframes)}
        4. Use ONLY these variables in your analysis: {method_selection.get('variables', [])}
        5. Remove NaN values with dropna() before any calculations
        6. Store ALL calculated metrics in the 'results' dictionary
        7. The LAST line of your code MUST be: print(json.dumps(results, default=str))
        8. Write ONLY executable Python code in the 'code' field
        9. NO comments explaining logic, NO markdown, NO descriptive text in the code
        10. **DO NOT filter data by date/time - the DataFrames are already filtered for the correct time period**
        11. **DO NOT use datetime.now() or any date-based filtering - work with all rows in the provided DataFrames**
    
        IMPORTANT NOTE ABOUT DATA:
        - All DataFrames have already been filtered for the appropriate time period
        - DO NOT apply additional date filters using datetime.now(), timedelta, or date comparisons
        - Use ALL rows present in each DataFrame
        - If you need to merge DataFrames, ensure date columns are properly formatted but don't filter by date
        
        Example structure for correlation analysis using actual dataframe names:
        df_sleep = get_sleep_data[['wakeup_count', 'total_sleep_time']].dropna()
        X = sm.add_constant(df_sleep['wakeup_count'])
        y = df_sleep['total_sleep_time']
        model = sm.OLS(y, X).fit()
        correlation = np.corrcoef(df_sleep['wakeup_count'], df_sleep['total_sleep_time'])[0, 1]
        results['correlation'] = float(correlation)
        results['r_squared'] = float(model.rsquared)
        results['p_value'] = float(model.pvalues[1])
        results['n_observations'] = int(len(df_sleep))
        print(json.dumps(results, default=str))
    
        Example for merging multiple dataframes:
        df_sleep = get_sleep_data[['data', 'wakeup_count']].dropna()
        df_kitchen = get_kitchen_data[['timestamp_picco', 'durata_attivita_minuti']].dropna()
        df_kitchen = df_kitchen.rename(columns={{'timestamp_picco': 'data'}})
        merged = pd.merge(df_sleep, df_kitchen, on='data', how='inner')
        X = sm.add_constant(merged['wakeup_count'])
        y = merged['durata_attivita_minuti']
        model = sm.OLS(y, X).fit()
        results['correlation'] = float(np.corrcoef(merged['wakeup_count'], merged['durata_attivita_minuti'])[0, 1])
        results['p_value'] = float(model.pvalues[1])
        print(json.dumps(results, default=str))
    
        Return:
        - imports: any additional imports needed (or empty string if none)
        - code: clean, executable Python code following the requirements above
    """
    # REFLECTION se ci sono errori precedenti
    if state["analysis_errors"]:
        last_error = state["analysis_errors"][-1]

        reflection_prompt = f"""
                /no thinking
                PREVIOUS ATTEMPT FAILED!
                
                Your previous code:
                {last_error['code'][:400]}
                
                Error received:
                {last_error['error'][:400]}
                
                The error indicates a problem with your code. Generate NEW, CORRECTED code that:
                SUbject to analyze
                {subject}
                1. Fixes the specific error mentioned above
                2. Is syntactically valid Python
                3. Contains NO descriptive text or comments in Italian/English within the code
                4. Uses ONLY the specified variables: {method_selection.get('variables', [])}
                5. Properly handles NaN values
                6. Follows all requirements listed above
                
                Generate corrected code now.
        """
        base_prompt += reflection_prompt

    try:

        structured_llm = llm.with_structured_output(StatisticalAnalysisCode)
        result = structured_llm.invoke([HumanMessage(content=base_prompt)],reasoning=False)

        print(f"✓ Structured output ricevuto")
        print(f"  Explanation: {result.explanation[:80]}...")

        # Estrai il codice (già separato dal testo grazie a Pydantic)
        analysis_code = result.code.strip()

        # Safety: rimuovi eventuali markdown residui
        analysis_code = re.sub(r'<think>.*?</think>\s*', '', analysis_code, flags=re.DOTALL)
        analysis_code = re.sub(r'```python\n?', '', analysis_code)
        analysis_code = re.sub(r'```\n?', '', analysis_code)
        import textwrap
        analysis_code = textwrap.dedent(analysis_code).strip()
        # Validazione minima
        if 'results' not in analysis_code or 'print(json.dumps' not in analysis_code:
            error_msg = "Generated code missing 'results' dict or print statement"
            state["analysis_errors"].append({
                "attempt": state["analysis_attempts"],
                "code": analysis_code,
                "error": error_msg
            })
            print(f"❌ {error_msg}")
            return state

        state["analysis_code"] = analysis_code
        state["analysis_imports"] = result.imports
        print(f"✓ Codice generato e validato (tentativo {state['analysis_attempts']})")

        # Debug preview
        preview = '\n'.join(analysis_code.split('\n')[:3])
        print(f"  Preview: {preview}...")

    except Exception as e:
        import traceback
        error_msg = f"Errore generazione codice: {str(e)}\n{traceback.format_exc()}"
        state["error"] = error_msg
        print(f"❌ {error_msg}")

    return state


def check_analysis_code_node(state: SleepAnalysisState) -> SleepAnalysisState:
    """
    Nodo 2.5: Esegue il codice di analisi e verifica se ci sono errori.
    Se ci sono errori, li salva per il reflection step.
    """
    print("\n[NODE 2.5] Verifica esecuzione codice analisi...")

    if state.get("error"):
        return state

    analysis_code = state["analysis_code"]
    raw_data = state["raw_data"]

    custom_imports = state.get("analysis_imports", "")

    try:
        # Prepara i dataframe in base alle chiavi presenti in raw_data
        dataframes_setup = []
        print(raw_data)
        for key, records in raw_data.items():
            df_name = key  # es. 'get_kitchen_data' o 'get_sleep_data'
            dataframes_setup.append(f"{df_name} = pd.DataFrame({records})")

            if records and len(records) > 0:
                first_record = records[0]
                # Cerca colonne di tipo Timestamp o con 'data', 'date', 'time' nel nome
                for col in first_record.keys():
                    if 'data' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
                        dataframes_setup.append(f"{df_name}['{col}'] = pd.to_datetime({df_name}['{col}'])")

        dataframes_code = '\n'.join(dataframes_setup)

        full_code = f"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import diagnostic, stattools
from statsmodels.tsa import seasonal, stattools as tsa_stattools
import json
{custom_imports}

# Crea i dataframe dai raw_data
{dataframes_code}

results = {{}}

{analysis_code}

if 'print(json.dumps(results' not in '''{analysis_code}''':
    print(json.dumps(results, default=str))
"""
        output = python_repl.run(full_code)

        if output and output.strip():
            output_lines = output.strip().split('\n')
            json_output = None

            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_output = line
                    break

            if json_output:
                results = json.loads(json_output)
                cleaned_results = {}

                for key, value in results.items():
                    if isinstance(value, str):
                        cleaned_value = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                        cleaned_value = ' '.join(cleaned_value.split())
                        cleaned_results[key] = cleaned_value
                    else:
                        cleaned_results[key] = value

                state["analysis_results"] = cleaned_results
                print(f"✓ Codice eseguito con successo: {len(cleaned_results)} metriche")
                return state
            else:
                # Nessun JSON trovato - salva errore
                error_msg = f"JSON non trovato nell'output. Output ricevuto: {output[:500]}"
                state["analysis_errors"].append({
                    "attempt": state["analysis_attempts"],
                    "code": analysis_code,
                    "error": error_msg
                })
                print(f"⚠️ Errore: {error_msg}")
                return state
        else:
            error_msg = "Il codice non ha prodotto output"
            state["analysis_errors"].append({
                "attempt": state["analysis_attempts"],
                "code": analysis_code,
                "error": error_msg
            })
            print(f"⚠️ {error_msg}")
            return state

    except Exception as e:
        # Cattura l'errore per reflection
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        state["analysis_errors"].append({
            "attempt": state["analysis_attempts"],
            "code": analysis_code,
            "error": error_msg
        })

        print(f"❌ Errore esecuzione: {str(e)}")
        return state

# ==================== NODE 3: PLOTLY PLOT GENERATION ====================
def plotly_plot_generation_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 3: Genera codice Python per creare un grafico Plotly.
    """
    print("\n[NODE 3] Generazione grafico Plotly...")

    if state.get("error"):
        return state

    query = state["query"]
    analysis_results = state["analysis_results"]
    raw_data = state["raw_data"]
    method_selection = state.get("statistical_method", {})
    visualization_type = method_selection.get("visualization_type", "bar")

    state["plot_attempts"] = state.get("plot_attempts", 0) + 1
    MAX_PLOT_ATTEMPTS = 3

    # Crea LLM con structured output
    structured_llm = llm.with_structured_output(PlotlyGraphCode)

    # Prepara informazioni sui dataframe disponibili
    available_dataframes = {}
    for key, records in raw_data.items():
        if records and len(records) > 0:
            columns = list(records[0].keys())
            available_dataframes[key] = columns

    dataframes_info = "\n".join([
        f"- {df_name}: {len(raw_data[df_name])} records, columns: {', '.join(cols)}"
        for df_name, cols in available_dataframes.items()
    ])

    base_prompt = f"""
/no thinking
Generate Python code to create a Plotly visualization.

Query: "{query}"

Analysis Details:
{json.dumps(method_selection, indent=2)}

Analysis Results:
{json.dumps(analysis_results, indent=2, default=str)}

Available DataFrames:
{dataframes_info}

Suggested visualization type: {visualization_type}

Libraries ALREADY imported (do NOT re-import):
- pandas as pd
- numpy as np
- plotly.graph_objects as go
- plotly.express as px
- json

Variables ALREADY available:
{', '.join([f'- {df_name}: pandas DataFrame' for df_name in available_dataframes.keys()])}
- fig: None (you need to create the plotly figure)

CRITICAL Requirements:
1. Create a Plotly figure using plotly.graph_objects or plotly.express
2. Use the appropriate DataFrame ({', '.join(available_dataframes.keys())})
3. Use ONLY the variables specified in the analysis: {method_selection.get('variables', [])}
4. Handle NaN values appropriately (dropna() if needed)
5. Store the figure in variable 'fig'
6. The LAST line MUST be: print(fig.to_json())
7. Add appropriate titles, axis labels, and hover tooltips in Italian
8. For correlations/scatter plots, add trendline if relevant
9. Use professional color schemes
10. NO comments in the code, ONLY executable Python

Example for correlation (using get_sleep_data):
df_clean = get_sleep_data[['total_sleep_time', 'hr_average']].dropna()
fig = px.scatter(df_clean, x='total_sleep_time', y='hr_average', 
                 trendline='ols', 
                 title='Correlazione tra Sonno e Battito Cardiaco',
                 labels={{'total_sleep_time': 'Durata Sonno (min)', 'hr_average': 'Battito Medio'}})
fig.update_traces(marker=dict(size=8, opacity=0.7))
print(fig.to_json())

Example for time series (using get_sleep_data):
fig = go.Figure()
fig.add_trace(go.Scatter(x=get_sleep_data['data'], y=get_sleep_data['total_sleep_time'], 
                         mode='lines+markers', name='Sonno totale'))
fig.update_layout(title='Andamento del sonno', 
                  xaxis_title='Data', 
                  yaxis_title='Durata (minuti)')
print(fig.to_json())

Example combining both datasets:
fig = go.Figure()
fig.add_trace(go.Scatter(x=get_sleep_data['data'], y=get_sleep_data['total_sleep_time'],
                         mode='lines+markers', name='Sonno'))
fig.add_trace(go.Scatter(x=get_kitchen_data['timestamp_picco'], y=get_kitchen_data['temperatura_max'],
                         mode='markers', name='Temperatura Cucina', yaxis='y2'))
fig.update_layout(
    title='Sonno vs Attività in Cucina',
    yaxis=dict(title='Sonno (min)'),
    yaxis2=dict(title='Temperatura (°C)', overlaying='y', side='right')
)
print(fig.to_json())

Generate the code now.
"""

    # REFLECTION se ci sono errori precedenti
    if state.get("plot_errors"):
        last_error = state["plot_errors"][-1]
        reflection_prompt = f"""

PREVIOUS ATTEMPT FAILED!

Your previous code:
{last_error['code'][:400]}

Error received:
{last_error['error'][:400]}

Generate CORRECTED code that:
1. Fixes the specific error
2. Uses the correct DataFrame names ({', '.join(available_dataframes.keys())})
3. Is valid Python syntax
4. Creates a proper Plotly figure
5. Ends with print(fig.to_json())
"""
        base_prompt += reflection_prompt

    try:
        # Genera codice con structured output
        result = structured_llm.invoke([HumanMessage(content=base_prompt)],reasoning=False)

        print(f"✓ Structured output ricevuto")
        plot_code = result.code.strip()
        plot_code = re.sub(r'```python\n?', '', plot_code)
        plot_code = re.sub(r'```\n?', '', plot_code)

        # Validazione minima
        if 'fig' not in plot_code or 'print(fig.to_json())' not in plot_code:
            error_msg = "Generated code missing 'fig' variable or print statement"
            if "plot_errors" not in state:
                state["plot_errors"] = []
            state["plot_errors"].append({
                "attempt": state["plot_attempts"],
                "code": plot_code,
                "error": error_msg
            })
            print(f"❌ {error_msg}")

            if state["plot_attempts"] >= MAX_PLOT_ATTEMPTS:
                state["error"] = f"Impossibile generare grafico dopo {MAX_PLOT_ATTEMPTS} tentativi"
            return state

        # Prepara i dataframe
        dataframes_setup = []
        for key, records in raw_data.items():
            df_name = key
            dataframes_setup.append(f"{df_name} = pd.DataFrame({records})")

            # Converti colonne temporali
            if records and len(records) > 0:
                first_record = records[0]
                for col in first_record.keys():
                    if 'data' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
                        dataframes_setup.append(f"{df_name}['{col}'] = pd.to_datetime({df_name}['{col}'])")

        dataframes_code = '\n'.join(dataframes_setup)

        full_code = f"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
{result.imports}

{dataframes_code}

fig = None

{plot_code}
"""

        output = python_repl.run(full_code)
        print("grafico outpu",output)
        if output and output.strip():
            # Estrai il JSON del grafico
            output_lines = output.strip().split('\n')
            json_output = None

            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith('{') and '"data"' in line:
                    json_output = line
                    break

            if json_output:
                fig_json = json.loads(json_output)
                state["plotly_figure"] = fig_json
                state.pop("plot_errors", None)
                print(f"✓ Grafico Plotly generato con successo!")
                return state
            else:
                error_msg = "JSON del grafico non trovato nell'output"
                if "plot_errors" not in state:
                    state["plot_errors"] = []
                state["plot_errors"].append({
                    "attempt": state["plot_attempts"],
                    "code": plot_code,
                    "error": error_msg
                })
                print(f"❌ {error_msg}")
        else:
            error_msg = "Il codice non ha prodotto output"
            if "plot_errors" not in state:
                state["plot_errors"] = []
            state["plot_errors"].append({
                "attempt": state["plot_attempts"],
                "code": plot_code,
                "error": error_msg
            })
            print(f"❌ {error_msg}")

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        if "plot_errors" not in state:
            state["plot_errors"] = []
        state["plot_errors"].append({
            "attempt": state["plot_attempts"],
            "code": plot_code if 'plot_code' in locals() else "N/A",
            "error": error_msg
        })
        print(f"❌ Errore esecuzione: {str(e)}")

    # Retry logic
    if state["plot_attempts"] < MAX_PLOT_ATTEMPTS:
        print(f"⟳ Riprovo generazione grafico (tentativo {state['plot_attempts'] + 1}/{MAX_PLOT_ATTEMPTS})")
        return plotly_plot_generation_node(state, llm)
    else:
        state["error"] = f"Impossibile generare grafico dopo {MAX_PLOT_ATTEMPTS} tentativi"

    return state


# ==================== GRAPH CREATION ====================
def create_sleep_analysis_chain() -> StateGraph:
    """Crea la chain LangGraph con reflection per l'analisi"""

    workflow = StateGraph(SleepAnalysisState)

    workflow.add_node("extract_data", lambda state: extract_sleep_data_node(state, oll_tool))
    workflow.add_node("select_method", lambda state: select_statistical_method_node(state, llm_code))
    workflow.add_node("analyze", lambda state: statistical_analysis_node(state, llm_code))
    workflow.add_node("check_code", lambda state: check_analysis_code_node(state))
    workflow.add_node("plot", lambda state: plotly_plot_generation_node(state, llm_code))

    workflow.set_entry_point("extract_data")

    def check_error_extract(state: SleepAnalysisState) -> Literal["select_method", "end"]:
        return "end" if state.get("error") else "select_method"

    def check_error_method(state: SleepAnalysisState) -> Literal["analyze", "end"]:
        return "end" if state.get("error") else "analyze"

    def after_analyze(state: SleepAnalysisState) -> Literal["check_code", "end"]:
        return "end" if state.get("error") else "check_code"

    MAX_ANALYSIS_ATTEMPTS = 3

    def after_check_code(state: SleepAnalysisState) -> Literal["analyze", "plot", "end"]:
        if state.get("error"):
            return "end"

        if state.get("analysis_results"):
            return "plot"

        if state["analysis_errors"] and state["analysis_attempts"] < MAX_ANALYSIS_ATTEMPTS:
            print(f"⟳ Riprovo analisi (tentativo {state['analysis_attempts'] + 1}/{MAX_ANALYSIS_ATTEMPTS})")
            return "analyze"

        state["error"] = f"Impossibile completare l'analisi dopo {MAX_ANALYSIS_ATTEMPTS} tentativi"
        return "end"

    workflow.add_conditional_edges("extract_data", check_error_extract,
                                   {"select_method": "select_method", "end": END})
    workflow.add_conditional_edges("select_method", check_error_method,
                                   {"analyze": "analyze", "end": END})
    workflow.add_conditional_edges("analyze", after_analyze,
                                   {"check_code": "check_code", "end": END})
    workflow.add_conditional_edges("check_code", after_check_code,
                                   {"analyze": "analyze", "plot": "plot", "end": END})

    # MODIFICATO: plot va direttamente a END
    workflow.add_edge("plot", END)

    return workflow.compile()
# ==================== MAIN EXECUTION ====================
def run_analysis(query: str):
    """Esegue l'intera pipeline"""
    print(f"\n{'=' * 60}")
    print(f"QUERY: {query}")
    print(f"{'=' * 60}")

    chain = create_sleep_analysis_chain()

    initial_state = {
        "query": query,
        "subject_id": 0,
        "period": "",
        "raw_data": {},
        "statistical_method": {},
        "analysis_code": "",
        "analysis_results": {},
        "analysis_imports": "",
        "analysis_errors": [],
        "analysis_attempts": 0,
        "final_response": "",
        "plotly_figure": {},
        "plot_attempts": 0,
        "plot_errors": [],
        "error": ""
    }

    final_state = chain.invoke(initial_state,reasoning=False)

    print(f"\n{'=' * 60}")
    print("RISULTATI")
    print(f"{'=' * 60}")

    if final_state.get("error"):
        print(f"\n❌ ERRORE: {final_state['error']}")
    else:
        if final_state.get("plotly_figure"):
            import plotly.graph_objects as go

            # Salva JSON
            json_file = f"sleep_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(final_state["plotly_figure"], f, indent=2)

            # Salva HTML interattivo
            fig = go.Figure(final_state["plotly_figure"])
            html_file = json_file.replace('.json', '.html')
            fig.write_html(html_file)

            print(f"\n✓ Grafico Plotly salvato in:")
            print(f"  - JSON: {json_file}")
            print(f"  - HTML: {html_file}")

    return final_state


def get_chain():
    """Funzione helper per Streamlit"""
    return create_sleep_analysis_chain()


# ==================== MAIN ====================
if __name__ == "__main__":
    queries = [
        "C'è qualche correlazione tra il numero di risvegli e la durata dell'attività in cucina per il soggetto 2 negli utlimi 1' giorni?"
    ]

    for query in queries:
        result = run_analysis(query)
        print("\n" + "=" * 60 + "\n")
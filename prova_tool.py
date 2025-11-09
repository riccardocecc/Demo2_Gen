from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage
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
from settings import llm_code
from tool import get_sleep_data, get_kitchen_data

python_repl = PythonREPLTool()


# ==================== STATE DEFINITION ====================
class StatisticalAnalysisCode(BaseModel):
    """Schema per il codice di analisi statistica"""
    explanation: str = Field(description="Brief explanation of statistical approach (1-2 sentences)")
    imports: str = Field(description="Additional Python imports if needed (empty string if none)")
    code: str = Field(description="Executable Python code that populates 'results' dict and ends with print(json.dumps(results, default=str))")


class PlotlyGraphCode(BaseModel):
    """Schema per il codice di generazione grafico Plotly"""
    explanation: str = Field(description="Brief explanation of visualization approach (1-2 sentences)")
    imports: str = Field(description="Additional Python imports if needed (empty string if none)")
    code: str = Field(description="Executable Python code that creates 'fig' plotly figure and ends with fig.to_json()")


class AnalysisState(TypedDict):
    """State che mantiene lo stato della conversazione attraverso i nodi"""
    query: str
    subject_id: int
    period: str
    raw_data: dict[str, any]
    statistical_method: dict
    analysis_code: str
    analysis_results: dict
    analysis_imports: str
    analysis_errors: list
    analysis_attempts: int
    domains_detected: list[str]
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
def extract_sleep_data_node(state: AnalysisState, llm: ChatGoogleGenerativeAI) -> AnalysisState:
    """
    Nodo 1: Estrae i dati dal CSV usando la query in linguaggio naturale.
    """
    print("\n[NODE 1] Estrazione dati dal CSV...")

    query = state["query"]
    llm_with_tools = llm.bind_tools([get_sleep_data, get_kitchen_data])

    extraction_prompt = f"""
    Sei un assistente sanitario che deve recuperare dati, hai a disposizione due tool:
    1) get_sleep_data: chiama questo tool se la query ri riferisce al sonno
    2) get_kitchen_data: chiama qusto toot se  la query si riferisce alla cucina
    IMPORTANTE!: se la query comprende entrambi i domini chiama ENTRAMBI i tool
    
    devi SOLO ESTRARRE I PARAMETRI NON CONSIDERARE TUTTO IL RESTO.
    

    Query dell'utente: "{query}"

    Analizza la query ed estrai:
    1. subject_id: ID del soggetto (numero intero). Se non specificato, usa 1 come default.
    2. period: periodo da analizzare in formato 'last_N_days' (es: 'last_30_days') oppure 'YYYY-MM-DD,YYYY-MM-DD'
       Se non specificato, usa 'last_30_days' come default.
    """
    result_tool: dict[str, any]={}
    domains_detected: list[str] = []
    try:
        response = llm_with_tools.invoke([HumanMessage(content=extraction_prompt)])
        for tool in response.tool_calls:
            subject_id = tool['args'].get('subject_id', 1)
            period = tool['args'].get('period', 'last_30_days')
            if tool['name'] == 'get_sleep_data':
                result = get_sleep_data.invoke({
                    'subject_id': subject_id,
                    'period': period
                })
                df_clean = process_sleep_data(result, verbose=False)
                df_clean['data'] = df_clean['data'].dt.strftime('%Y-%m-%d')
                cleaned_records = df_clean.to_dict('records')
                result_tool[tool['name']] = cleaned_records
                domains_detected.append('sleep')
            elif tool['name'] == 'get_kitchen_data':
                result = get_kitchen_data.invoke({
                    'subject_id': subject_id,
                    'period': period
                })
                df_clean = process_kitchen_data(result, verbose=False)
                df_clean['timestamp_picco'] = df_clean['timestamp_picco'].dt.strftime('%Y-%m-%d')
                if 'start_time_attivita' in df_clean.columns:
                    df_clean['start_time_attivita'] = df_clean['start_time_attivita'].dt.strftime('%H:%M:%S')

                if 'end_time_attivita' in df_clean.columns:
                    df_clean['end_time_attivita'] = df_clean['end_time_attivita'].dt.strftime('%H:%M:%S')

                cleaned_records = df_clean.to_dict('records')
                result_tool[tool['name']] = cleaned_records
                domains_detected.append('kitchen')


        print(f"✓ Estratti parametri - soggetto ID: {subject_id}, period: {period}")

        state["subject_id"] = subject_id
        state["period"] = period
        state["raw_data"] = result_tool
        state["domains_detected"] = domains_detected

        print(f"✓ Dati pronti per l'analisi: {len(result['records'])} record puliti")

    except Exception as e:
        state["error"] = f"Errore nell'estrazione dati: {str(e)}"
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    return state


# ==================== NODE 1.5: STATISTICAL METHOD SELECTION ====================
def select_statistical_method_node(state: AnalysisState, llm: ChatGoogleGenerativeAI) -> AnalysisState:
    """
    Nodo 1.5: Analizza la query e determina quale analisi statistica è più appropriata.
    """
    print("\n[NODE 1.5] Selezione metodo statistico...")

    if state.get("error"):
        return state

    query = state["query"]
    domains_detected = state.get("domains_detected", [])
    available_columns = domain_registry.get_available_columns_for_domains(domains_detected)
    print(f"✓ Colonne disponibili per i domini {domains_detected}: {available_columns}")

    selection_prompt = f"""
    Sei un esperto statistico specializzato nell'analisi dei dati sonno e cucina.

    Query dell'utente: "{query}"

    Dati disponibili (campi per ogni record):
    {available_columns}

    Numero di record disponibili: {len(available_columns)}

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
        "considerations": "eventuali note importanti"
    }}

    Rispondi SOLO con il JSON, senza markdown o spiegazioni aggiuntive.
    """

    try:
        response = llm.invoke([HumanMessage(content=selection_prompt)])
        response_text = response.content.strip()
        response_text = re.sub(r'```json\n?', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)

        method_selection = json.loads(response_text)
        state["statistical_method"] = method_selection

        print("statistical",state["statistical_method"])

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
def statistical_analysis_node(state: AnalysisState, llm: ChatGoogleGenerativeAI) -> AnalysisState:
    """
    Nodo 2: Genera codice Python per analisi statistica usando Pydantic structured output.
    """
    print(f"\n[NODE 2] Analisi statistica - Tentativo {state['analysis_attempts'] + 1}...")

    if state.get("error"):
        return state

    query = state["query"]
    raw_data = state["raw_data"]
    print("raw_data",raw_data)
    method_selection = state.get("statistical_method", {})

    state["analysis_attempts"] += 1

    # Crea LLM con structured output
    structured_llm = llm.with_structured_output(StatisticalAnalysisCode)
    available_columns = domain_registry.get_available_columns_for_domains(state.get("domains_detected", []))
    # Base prompt
    base_prompt = f"""
Generate Python code for statistical analysis using statsmodels.

Query: "{query}"

Analysis Details:
{method_selection}


Available columns:{available_columns}

Libraries ALREADY imported (do NOT re-import these):
- pandas as pd
- numpy as np
- statsmodels.api as sm
- statsmodels.formula.api as smf
- json

Variables ALREADY available in execution context:
- df: pandas DataFrame with cleaned data
- results: empty dictionary {{}} to populate with your analysis results

CRITICAL Requirements:
1. Use ONLY statsmodels for statistical calculations (NO scipy.stats, NO sklearn)
2. Use ONLY these variables in your analysis: {method_selection.get('variables', [])}
3. Remove NaN values with dropna() before any calculations
4. Store ALL calculated metrics in the 'results' dictionary
5. The LAST line of your code MUST be: print(json.dumps(results, default=str))
6. Write ONLY executable Python code in the 'code' field
7. NO comments explaining logic, NO markdown, NO descriptive text in the code

Example structure for correlation analysis:
df_clean = df[['variable1', 'variable2']].dropna()
X = sm.add_constant(df_clean['variable1'])
y = df_clean['variable2']
model = sm.OLS(y, X).fit()
correlation = np.corrcoef(df_clean['variable1'], df_clean['variable2'])[0, 1]
results['correlation'] = float(correlation)
results['r_squared'] = float(model.rsquared)
results['p_value'] = float(model.pvalues[1])
results['n_observations'] = int(len(df_clean))
print(json.dumps(results, default=str))

Return:
- explanation: 1-2 sentences describing your statistical approach
- imports: any additional imports needed (or empty string if none)
- code: clean, executable Python code following the requirements above
"""

    # REFLECTION se ci sono errori precedenti
    if state["analysis_errors"]:
        last_error = state["analysis_errors"][-1]

        reflection_prompt = f"""

PREVIOUS ATTEMPT FAILED!

Your previous code:
{last_error['code'][:400]}

Error received:
{last_error['error'][:400]}

The error indicates a problem with your code. Generate NEW, CORRECTED code that:
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
        # Invoca LLM con structured output
        result = structured_llm.invoke([HumanMessage(content=base_prompt)])

        print(f"✓ Structured output ricevuto")
        print(f"  Explanation: {result.explanation[:80]}...")

        # Estrai il codice (già separato dal testo grazie a Pydantic)
        analysis_code = result.code.strip()

        # Safety: rimuovi eventuali markdown residui
        analysis_code = re.sub(r'```python\n?', '', analysis_code)
        analysis_code = re.sub(r'```\n?', '', analysis_code)

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


def check_analysis_code_node(state: AnalysisState) -> AnalysisState:
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
        full_code = f"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import diagnostic, stattools
from statsmodels.tsa import seasonal, stattools as tsa_stattools
import json
{custom_imports}
df = pd.DataFrame({raw_data['records']})
df['data'] = pd.to_datetime(df['data'])
results = {{}}

{analysis_code}

if 'print(json.dumps(results' not in '''{analysis_code}''':
    print(json.dumps(results, default=str))
"""

        # Esegui il codice
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
def plotly_plot_generation_node(state: AnalysisState, llm: ChatGoogleGenerativeAI) -> AnalysisState:
    """
    Nodo 3: Genera codice Python per creare un grafico Plotly.
    """
    print("\n[NODE 3] Generazione grafico Plotly...")

    if state.get("error"):
        return state

    query = state["query"]
    analysis_results = state["analysis_results"]
    raw_data = state["raw_data"]
    domains_detected = state.get("domains_detected", [])
    method_selection = state.get("statistical_method", {})
    visualization_type = method_selection.get("visualization_type", "bar")

    state["plot_attempts"] = state.get("plot_attempts", 0) + 1
    MAX_PLOT_ATTEMPTS = 3

    # PREPARAZIONE DEI DATI PER IL PROMPT
    # Conta i record totali e costruisci descrizione dei domini
    total_records = sum(len(data) for data in raw_data.values())

    domains_info = []
    for domain_name in domains_detected:
        domain_config = domain_registry.get_domain(domain_name)
        if domain_config:
            tool_key = f'get_{domain_name}_data'
            if tool_key in raw_data:
                num_records = len(raw_data[tool_key])
                columns = domain_config.get_available_columns()
                domains_info.append(f"- {domain_name}: {num_records} records, colonne: {', '.join(columns)}")

    domains_description = "\n".join(domains_info)

    # Crea LLM con structured output
    structured_llm = llm.with_structured_output(PlotlyGraphCode)

    base_prompt = f"""
Generate Python code to create a Plotly visualization.

Query: "{query}"

Analysis Details:
{json.dumps(method_selection, indent=2)}

Analysis Results:
{json.dumps(analysis_results, indent=2, default=str)}

Domains and Data Available:
{domains_description}

Total records: {total_records}

Suggested visualization type: {visualization_type}

Libraries ALREADY imported (do NOT re-import):
- pandas as pd
- numpy as np
- plotly.graph_objects as go
- plotly.express as px
- json

Variables ALREADY available:
- raw_data: dict with keys {list(raw_data.keys())}
  Each key contains a list of records for that domain
- df_sleep: pandas DataFrame with sleep data (if available)
- df_kitchen: pandas DataFrame with kitchen data (if available)
- fig: None (you need to create the plotly figure)

CRITICAL Requirements:
1. Create a Plotly figure using plotly.graph_objects or plotly.express
2. Use the appropriate DataFrame(s) based on domains involved
3. Use ONLY the variables specified in the analysis: {method_selection.get('variables', [])}
4. Handle NaN values appropriately (dropna() if needed)
5. Store the figure in variable 'fig'
6. The LAST line MUST be: print(fig.to_json())
7. Add appropriate titles, axis labels, and hover tooltips in Italian
8. For correlations/scatter plots, add trendline if relevant
9. Use professional color schemes
10. NO comments in the code, ONLY executable Python

Example for single domain (sleep):
df_clean = df_sleep[['data', 'total_sleep_time']].dropna()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_clean['data'], y=df_clean['total_sleep_time'], 
                         mode='lines+markers', name='Sonno totale'))
fig.update_layout(title='Andamento del sonno', 
                  xaxis_title='Data', 
                  yaxis_title='Durata (minuti)')
print(fig.to_json())

Example for cross-domain analysis:
df_sleep_clean = df_sleep[['data', 'total_sleep_time']].dropna()
df_kitchen_clean = df_kitchen[['timestamp_picco', 'durata_attivita_minuti']].dropna()
df_kitchen_clean = df_kitchen_clean.rename(columns={{'timestamp_picco': 'data'}})
df_kitchen_agg = df_kitchen_clean.groupby('data')['durata_attivita_minuti'].sum().reset_index()
df_merged = pd.merge(df_sleep_clean, df_kitchen_agg, on='data', how='inner')
fig = px.scatter(df_merged, x='durata_attivita_minuti', y='total_sleep_time',
                 trendline='ols',
                 title='Relazione tra Attività in Cucina e Sonno',
                 labels={{'durata_attivita_minuti': 'Tempo in cucina (min)', 
                         'total_sleep_time': 'Durata sonno (min)'}})
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
2. Is valid Python syntax
3. Creates a proper Plotly figure
4. Uses the correct DataFrame names (df_sleep, df_kitchen)
5. Ends with print(fig.to_json())
"""
        base_prompt += reflection_prompt

    try:
        # Genera codice con structured output
        result = structured_llm.invoke([HumanMessage(content=base_prompt)])

        print(f"✓ Structured output ricevuto")
        print(f"  Explanation: {result.explanation[:80]}...")

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

        # PREPARA I DATAFRAMES PER L'ESECUZIONE
        dataframe_setup = []

        if 'get_sleep_data' in raw_data:
            dataframe_setup.append(f"""
df_sleep = pd.DataFrame({raw_data['get_sleep_data']})
df_sleep['data'] = pd.to_datetime(df_sleep['data'])
""")

        if 'get_kitchen_data' in raw_data:
            dataframe_setup.append(f"""
df_kitchen = pd.DataFrame({raw_data['get_kitchen_data']})
df_kitchen['timestamp_picco'] = pd.to_datetime(df_kitchen['timestamp_picco'])
""")

        full_code = f"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
{result.imports}

raw_data = {raw_data}

{''.join(dataframe_setup)}

fig = None

{plot_code}
"""

        output = python_repl.run(full_code)

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

# ==================== NODE 4: NATURAL LANGUAGE RESPONSE ====================
def generate_response_node(state: AnalysisState, llm: ChatGoogleGenerativeAI) -> AnalysisState:
    """
    Nodo 4: Genera risposta in linguaggio naturale per medici.
    """
    print("\n[NODE 4] Generazione risposta...")

    if state.get("error"):
        return state

    query = state["query"]
    analysis_results = state["analysis_results"]
    method_selection = state.get("statistical_method", {})
    subject_id = state["subject_id"]
    period = state["period"]
    num_records = len(state["raw_data"]["records"])

    response_prompt = f"""
Sei un assistente medico che spiega risultati statistici a un medico.

QUERY: "{query}"

CONTESTO:
- Paziente ID: {subject_id}
- Periodo: {period}
- Notti analizzate: {num_records}
- Tipo analisi: {method_selection.get('analysis_type', 'N/A')}

RISULTATI:
{json.dumps(analysis_results, indent=2, default=str)}

ISTRUZIONI:
1. Rispondi DIRETTAMENTE alla domanda
2. Linguaggio clinico ma accessibile
3. Struttura: risposta diretta (2-3 frasi)
4. Traduci statistiche in termini comprensibili
5. Max 150 parole
6. NO markdown, NO elenchi puntati
7. Paragrafi continui

Rispondi SOLO con il testo.
"""

    try:
        response = llm.invoke([HumanMessage(content=response_prompt)])
        final_response = response.content.strip()
        state["final_response"] = final_response
        print(f"✓ Risposta generata ({len(final_response)} caratteri)")

    except Exception as e:
        state["error"] = f"Errore generazione risposta: {str(e)}"

    return state


# ==================== GRAPH CREATION ====================
def create_sleep_analysis_chain() -> StateGraph:
    """Crea la chain LangGraph con reflection per l'analisi"""

    workflow = StateGraph(AnalysisState)

    workflow.add_node("extract_data", lambda state: extract_sleep_data_node(state, llm_code))
    workflow.add_node("select_method", lambda state: select_statistical_method_node(state, llm_code))
    workflow.add_node("analyze", lambda state: statistical_analysis_node(state, llm_code))
    workflow.add_node("check_code", lambda state: check_analysis_code_node(state))
    workflow.add_node("plot", lambda state: plotly_plot_generation_node(state, llm_code))
    workflow.add_node("respond", lambda state: generate_response_node(state, llm_code))

    workflow.set_entry_point("extract_data")

    def check_error_extract(state: AnalysisState) -> Literal["select_method", "end"]:
        return "end" if state.get("error") else "select_method"

    def check_error_method(state: AnalysisState) -> Literal["analyze", "end"]:
        return "end" if state.get("error") else "analyze"

    def after_analyze(state: AnalysisState) -> Literal["check_code", "end"]:
        return "end" if state.get("error") else "check_code"

    MAX_ANALYSIS_ATTEMPTS = 3

    def after_check_code(state: AnalysisState) -> Literal["analyze", "plot", "end"]:
        if state.get("error"):
            return "end"

        if state.get("analysis_results"):
            return "plot"

        if state["analysis_errors"] and state["analysis_attempts"] < MAX_ANALYSIS_ATTEMPTS:
            print(f"⟳ Riprovo analisi (tentativo {state['analysis_attempts'] + 1}/{MAX_ANALYSIS_ATTEMPTS})")
            return "analyze"

        state["error"] = f"Impossibile completare l'analisi dopo {MAX_ANALYSIS_ATTEMPTS} tentativi"
        return "end"

    def check_error_plot(state: AnalysisState) -> Literal["respond", "end"]:
        return "end" if state.get("error") else "respond"

    workflow.add_conditional_edges("extract_data", check_error_extract,
                                   {"select_method": "select_method", "end": END})
    workflow.add_conditional_edges("select_method", check_error_method,
                                   {"analyze": "analyze", "end": END})
    workflow.add_conditional_edges("analyze", after_analyze,
                                   {"check_code": "check_code", "end": END})
    workflow.add_conditional_edges("check_code", after_check_code,
                                   {"analyze": "analyze", "plot": "plot", "end": END})
    workflow.add_conditional_edges("plot", check_error_plot,
                                   {"respond": "respond", "end": END})

    workflow.add_edge("respond", END)

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

    final_state = chain.invoke(initial_state)

    print(f"\n{'=' * 60}")
    print("RISULTATI")
    print(f"{'=' * 60}")

    if final_state.get("error"):
        print(f"\n❌ ERRORE: {final_state['error']}")
    else:
        print(f"\n✓ Subject ID: {final_state['subject_id']}")
        print(f"✓ Period: {final_state['period']}")
        print(f"✓ Records: {len(final_state['raw_data']['records'])}")

        print(f"\n{'=' * 60}")
        print("RISPOSTA CLINICA")
        print(f"{'=' * 60}")
        print(final_state['final_response'])
        print(f"{'=' * 60}")

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
        "C'è correlazione tra il numero di risvegli e l'utilizzo della cucina per soggetto 2 negli utlmi 10 giorni"
    ]

    for query in queries:
        result = run_analysis(query)
        print("\n" + "=" * 60 + "\n")
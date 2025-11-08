from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL

from settings import llm_code
from tool import get_sleep_data

python_repl = PythonREPLTool()


# ==================== STATE DEFINITION ====================
class StatisticalAnalysisCode(BaseModel):
    """Schema per il codice di analisi statistica"""
    explanation: str = Field(description="Brief explanation of statistical approach (1-2 sentences)")
    imports: str = Field(description="Additional Python imports if needed (empty string if none)")
    code: str = Field(description="Executable Python code that populates 'results' dict and ends with print(json.dumps(results, default=str))")

class SleepAnalysisState(TypedDict):
    """State che mantiene lo stato della conversazione attraverso i nodi"""
    query: str
    subject_id: int
    period: str
    raw_data: dict
    data_sources: list
    statistical_method: dict
    analysis_code: str
    analysis_results: dict
    analysis_errors: list  # NUOVO: storico errori
    analysis_attempts: int  # NUOVO: contatore tentativi
    vega_spec: dict  # CAMBIATO: dizionario Vega-Lite invece di codice
    plot_html: str
    messages: list
    error: str
    final_response: str


# ==================== TOOL DEFINITION ====================
class SleepDataResult(TypedDict):
    subject_id: int
    period: str
    records: list


class ErrorResult(TypedDict):
    error: str


SLEEP_DATA_PATH = 'sleep_data.csv'


# ==================== NODE 1: DATA EXTRACTION ====================
def extract_sleep_data_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 1: Estrae i dati dal CSV usando la query in linguaggio naturale.
    """
    print("\n[NODE 1] Estrazione dati dal CSV...")

    query = state["query"]
    llm_with_tools = llm.bind_tools([get_sleep_data])

    extraction_prompt = f"""
    Sei un assistente che deve recuperare dati del sonno usando il tool get_sleep_data, devi SOLO ESTRARRE I PARAMETRI NON CONSIDERARE TUTTO IL RESTO.

    Query dell'utente: "{query}"

    Analizza la query ed estrai:
    1. subject_id: ID del soggetto (numero intero). Se non specificato, usa 1 come default.
    2. period: periodo da analizzare in formato 'last_N_days' (es: 'last_30_days') oppure 'YYYY-MM-DD,YYYY-MM-DD'
       Se non specificato, usa 'last_30_days' come default.

    Chiama il tool get_sleep_data con i parametri appropriati per recuperare i dati.
    """

    try:
        response = llm_with_tools.invoke([HumanMessage(content=extraction_prompt)])

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            subject_id = tool_call['args'].get('subject_id', 1)
            period = tool_call['args'].get('period', 'last_30_days')

            print(f"✓ Estratti parametri - soggetto ID: {subject_id}, period: {period}")

            result = get_sleep_data.invoke({
                'subject_id': subject_id,
                'period': period
            })

            if isinstance(result, dict) and 'error' in result:
                state["error"] = result['error']
                return state

            # Pulizia dati
            df = pd.DataFrame(result['records'])
            print(f"✓ Dati estratti: {len(df)} record")
            print(f"  Pulizia dati in corso...")

            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'subject_id' in numeric_cols:
                numeric_cols.remove('subject_id')

            for col in numeric_cols:
                df.loc[df[col] < 0, col] = np.nan
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            initial_rows = len(df)
            df = df.dropna(subset=['data'])
            if len(df) < initial_rows:
                print(f"Rimosse {initial_rows - len(df)} righe con date mancanti")

            initial_rows = len(df)
            df = df.drop_duplicates(subset=['data', 'subject_id'], keep='first')
            if len(df) < initial_rows:
                print(f"Rimosse {initial_rows - len(df)} righe duplicate")

            df = df.sort_values('data').reset_index(drop=True)

            outlier_cols = ['total_sleep_time', 'hr_average', 'rr_average', 'wakeup_count']
            for col in outlier_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > 0:
                        print(f"  ⚠️ Rilevati {outliers} outliers in '{col}' (impostati a NaN)")
                        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan

            print(f"✓ Pulizia completata: {len(df)} record validi")

            df['data'] = df['data'].dt.strftime('%Y-%m-%d')
            cleaned_records = df.to_dict('records')
            result['records'] = cleaned_records

            state["subject_id"] = subject_id
            state["period"] = period
            state["raw_data"] = result

            print(f"✓ Dati pronti per l'analisi: {len(result['records'])} record puliti")

        else:
            print("⚠️ LLM non ha chiamato il tool, uso fallback con regex...")
            subject_id = 1
            period = "last_30_days"

            subject_match = re.search(r'soggetto[:\s]+(\d+)|subject[:\s]+(\d+)|id[:\s]+(\d+)', query, re.IGNORECASE)
            if subject_match:
                subject_id = int([g for g in subject_match.groups() if g][0])

            if 'ultimi' in query.lower() or 'last' in query.lower():
                days_match = re.search(r'(\d+)\s*giorni|(\d+)\s*days', query, re.IGNORECASE)
                if days_match:
                    days = int([g for g in days_match.groups() if g][0])
                    period = f"last_{days}_days"
            elif re.search(r'\d{4}-\d{2}-\d{2}', query):
                dates = re.findall(r'\d{4}-\d{2}-\d{2}', query)
                if len(dates) >= 2:
                    period = f"{dates[0]},{dates[1]}"

            result = get_sleep_data.invoke({
                'subject_id': subject_id,
                'period': period
            })

            if isinstance(result, dict) and 'error' in result:
                state["error"] = result['error']
                return state

            # Stessa pulizia dati del branch precedente
            df = pd.DataFrame(result['records'])
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            # ... (ripeti la pulizia come sopra)
            df['data'] = df['data'].dt.strftime('%Y-%m-%d')
            cleaned_records = df.to_dict('records')
            result['records'] = cleaned_records

            state["subject_id"] = subject_id
            state["period"] = period
            state["raw_data"] = result

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

    query = state["query"]
    raw_data = state["raw_data"]
    print("Numero di record disponibili:", len(raw_data['records']))

    selection_prompt = f"""
    Sei un esperto statistico specializzato nell'analisi dei dati del sonno.

    Query dell'utente: "{query}"

    Dati disponibili (campi per ogni record):
    - data: data della registrazione
    - total_sleep_time: durata totale del sonno in minuti
    - rem_sleep_duration: durata fase REM in minuti
    - deep_sleep_duration: durata sonno profondo in minuti
    - light_sleep_duration: durata sonno leggero in minuti
    - wakeup_count: numero di risvegli per notte
    - out_of_bed_count: numero di uscite dal letto
    - hr_average: frequenza cardiaca media (bpm)
    - rr_average: frequenza respiratoria media (respiri/min)

    Numero di record disponibili: {len(raw_data['records'])}

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
        "analysis_goal": "obiettivo principale dell'analisi",
        "analysis_type": "tipo di analisi (es: correlation, proportion, descriptive, trend, comparison, etc.)",
        "variables": ["lista", "delle", "colonne", "da", "analizzare"],
        "statistical_methods": ["metodo"],
        "calculations_needed": {{
            "metric1": "descrizione calcolo",
            "metric2": "descrizione calcolo"
        }},
        "expected_outputs": ["output1", "output2"],
        "visualization_type": "tipo di grafico suggerito",
        "reasoning": "spiegazione dettagliata della scelta",
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
            "reasoning": "Fallback",
            "considerations": ""
        }
        state["statistical_method"] = method_selection

    except Exception as e:
        state["error"] = f"Errore selezione metodo: {str(e)}"

    return state


# ==================== NODE 2: STATISTICAL ANALYSIS WITH REFLECTION ====================
# ==================== NODE 2: STATISTICAL ANALYSIS WITH STRUCTURED OUTPUT ====================
def statistical_analysis_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 2: Genera codice Python per analisi statistica usando Pydantic structured output.
    """
    print(f"\n[NODE 2] Analisi statistica - Tentativo {state['analysis_attempts'] + 1}...")

    if state.get("error"):
        return state

    query = state["query"]
    raw_data = state["raw_data"]
    method_selection = state.get("statistical_method", {})

    state["analysis_attempts"] += 1

    # Crea LLM con structured output
    structured_llm = llm.with_structured_output(StatisticalAnalysisCode)

    # Base prompt
    base_prompt = f"""
Generate Python code for statistical analysis using statsmodels.

Query: "{query}"

Analysis Details:
- Type: {method_selection.get('analysis_type', '')}
- Variables to use: {method_selection.get('variables', [])}
- Goal: {method_selection.get('analysis_goal', '')}

Data: {len(raw_data['records'])} records
Available columns: wakeup_count, hr_average, total_sleep_time, rem_sleep_duration, 
deep_sleep_duration, light_sleep_duration, out_of_bed_count, rr_average, data, subject_id

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
            print(f"⚠️ {error_msg}")
            return state

        state["analysis_code"] = analysis_code
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
# ==================== NODE 2.5: CODE EXECUTION CHECK ====================
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

    try:
        df = pd.DataFrame(raw_data['records'])
        df['data'] = pd.to_datetime(df['data'])

        full_code = f"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import diagnostic, stattools
from statsmodels.tsa import seasonal, stattools as tsa_stattools
import json

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



# ==================== NODE 3: VEGA-LITE PLOT GENERATION ====================
def vega_lite_plot_generation_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 3: Genera una specifica Vega-Lite JSON per visualizzare i dati.
    """
    print("\n[NODE 3] Generazione grafico Vega-Lite...")

    if state.get("error"):
        return state

    query = state["query"]
    analysis_results = state["analysis_results"]
    raw_data = state["raw_data"]
    method_selection = state.get("statistical_method", {})
    visualization_type = method_selection.get("visualization_type", "bar")

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"\nTentativo {attempt}/{max_attempts}")

        try:
            vega_prompt = create_vega_prompt(
                query=query,
                visualization_type=visualization_type,
                analysis_results=analysis_results,
                raw_data=raw_data,
                method_selection=method_selection,
                previous_error=state.get("last_vega_error"),
                previous_spec=state.get("vega_spec")
            )

            response = llm.invoke([HumanMessage(content=vega_prompt)])
            vega_text = response.content.strip()

            # Pulisci markdown
            vega_text = re.sub(r'```json\n?', '', vega_text)
            vega_text = re.sub(r'```\n?', '', vega_text)

            # Parse JSON
            vega_spec = json.loads(vega_text)

            # Valida la spec
            validation_error = validate_vega_spec(vega_spec)
            if validation_error:
                print(f"  ⚠️ Validazione fallita: {validation_error}")
                state["last_vega_error"] = validation_error
                continue

            # Aggiungi i dati alla spec
            df = pd.DataFrame(raw_data['records'])
            df['data'] = pd.to_datetime(df['data']).dt.strftime('%Y-%m-%d')

            # Rimuovi NaN per Vega-Lite
            df = df.replace({np.nan: None})

            vega_spec["data"] = {"values": df.to_dict('records')}

            print("grafico",vega_spec)
            state["vega_spec"] = vega_spec
            state["plot_html"] = ""
            state.pop("last_vega_error", None)
            print("✓ Grafico Vega-Lite generato con successo!")
            return state

        except json.JSONDecodeError as e:
            error_msg = f"JSON non valido: {str(e)}"
            print(f"  ❌ {error_msg}")
            state["last_vega_error"] = error_msg

        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Errore: {error_msg[:200]}")
            state["last_vega_error"] = error_msg

            if attempt == max_attempts:
                state[
                    "error"] = f"Impossibile generare grafico dopo {max_attempts} tentativi.\nUltimo errore: {error_msg}"
                print(f"\n❌ ERRORE FINALE")

    return state


def create_vega_prompt(query: str, visualization_type: str, analysis_results: dict,
                       raw_data: dict, method_selection: dict,
                       previous_error: str = None, previous_spec: dict = None) -> str:
    """Crea il prompt per generare la specifica Vega-Lite"""
    print("Tipo di visualizazoine",visualization_type )
    base_prompt = f"""
Sei un esperto di visualizzazione dati con Vega-Lite.

Query: "{query}"

RISULTATI ANALISI:
{json.dumps(analysis_results, indent=2, default=str)}

TIPO VISUALIZZAZIONE: {visualization_type}

VARIABILI DA method_selection: {method_selection.get('variables', [])}

CAMPI DISPONIBILI NEI DATI:
- data (datetime in formato YYYY-MM-DD)
- total_sleep_time, rem_sleep_duration, deep_sleep_duration, light_sleep_duration
- wakeup_count, out_of_bed_count, hr_average, rr_average, subject_id

REQUISITI:
1. Crea una specifica Vega-Lite COMPLETA in formato JSON
2. USA il campo "data" per l'asse temporale (tipo "temporal")
3. Usa le variabili specificate in method_selection['variables']
4. NON includere i dati nel JSON (verranno aggiunti automaticamente)
5. Includi title, axes labels, tooltips appropriati
6. Usa mark types validi: point, line, bar, area, circle, square, tick, rule, arc

STRUTTURA RICHIESTA:
{{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "title": "Titolo descrittivo",
  "mark": {{"type": "point"}},
  "encoding": {{
    "x": {{"field": "data", "type": "temporal", "title": "Data"}},
    "y": {{"field": "nome_campo", "type": "quantitative", "title": "Label"}},
    "tooltip": [...]
  }},
  "width": 600,
  "height": 400
}}

ERRORI DA EVITARE:
- NON inventare campi che non esistono
- NON includere "data": {{"values": ...}} nel JSON
- USA mark types validi (non "scatter", usa "point")
- Per correlazioni usa mark "point" con encoding x e y
- Per trend temporali usa mark "line"

Rispondi SOLO con il JSON della specifica Vega-Lite, senza markdown o spiegazioni.
"""

    if previous_error and previous_spec:
        feedback = f"""

ERRORE NEL TENTATIVO PRECEDENTE:
{previous_error}

SPECIFICA CHE HA CAUSATO L'ERRORE:
{json.dumps(previous_spec, indent=2)}

CORREGGI L'ERRORE E RIGENERA LA SPECIFICA.
"""
        base_prompt += feedback

    return base_prompt


def validate_vega_spec(spec: dict) -> str:
    """Valida una specifica Vega-Lite. Ritorna None se valida, messaggio di errore altrimenti."""

    required_fields = ["mark", "encoding"]
    for field in required_fields:
        if field not in spec:
            return f"Campo obbligatorio mancante: {field}"

    if "data" in spec and "values" in spec["data"]:
        return "Non includere 'data.values' nella spec (verrà aggiunto automaticamente)"

    valid_marks = ["arc", "area", "bar", "circle", "line", "point", "rect", "rule", "square", "text", "tick", "trail"]
    mark_type = spec["mark"]
    if isinstance(mark_type, dict):
        mark_type = mark_type.get("type", "")

    if mark_type not in valid_marks:
        return f"Mark type '{mark_type}' non valido. Usa uno di: {', '.join(valid_marks)}"

    return None



# ==================== NODE 4: NATURAL LANGUAGE RESPONSE ====================
def generate_response_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
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


def create_sleep_analysis_chain() -> StateGraph:
    """Crea la chain LangGraph con reflection per l'analisi"""

    workflow = StateGraph(SleepAnalysisState)

    workflow.add_node("extract_data", lambda state: extract_sleep_data_node(state, llm_code))
    workflow.add_node("select_method", lambda state: select_statistical_method_node(state, llm_code))
    workflow.add_node("analyze", lambda state: statistical_analysis_node(state, llm_code))
    workflow.add_node("check_code", lambda state: check_analysis_code_node(state))  # NUOVO
    workflow.add_node("plot", lambda state: vega_lite_plot_generation_node(state, llm_code))
    workflow.add_node("respond", lambda state: generate_response_node(state, llm_code))

    workflow.set_entry_point("extract_data")

    def check_error_extract(state: SleepAnalysisState) -> Literal["select_method", "end"]:
        return "end" if state.get("error") else "select_method"

    def check_error_method(state: SleepAnalysisState) -> Literal["analyze", "end"]:
        return "end" if state.get("error") else "analyze"

    # NUOVO: Conditional edge dopo analyze → check_code
    def after_analyze(state: SleepAnalysisState) -> Literal["check_code", "end"]:
        return "end" if state.get("error") else "check_code"

    # NUOVO: Conditional edge dopo check_code
    MAX_ANALYSIS_ATTEMPTS = 3

    def after_check_code(state: SleepAnalysisState) -> Literal["analyze", "plot", "end"]:
        # Se c'è un errore fatale, termina
        if state.get("error"):
            return "end"

        # Se ci sono risultati, procedi con il plot
        if state.get("analysis_results"):
            return "plot"

        # Se ci sono errori ma non abbiamo superato i tentativi max, riprova
        if state["analysis_errors"] and state["analysis_attempts"] < MAX_ANALYSIS_ATTEMPTS:
            print(f"⟳ Riprovo analisi (tentativo {state['analysis_attempts'] + 1}/{MAX_ANALYSIS_ATTEMPTS})")
            return "analyze"

        # Superati i tentativi max, errore fatale
        state["error"] = f"Impossibile completare l'analisi dopo {MAX_ANALYSIS_ATTEMPTS} tentativi"
        return "end"

    def check_error_plot(state: SleepAnalysisState) -> Literal["respond", "end"]:
        return "end" if state.get("error") else "respond"

    # Edges
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

# ==================== UTILITY FUNCTIONS ====================
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
        "data_sources": [],
        "statistical_method": {},
        "analysis_code": "",
        "analysis_results": {},
        "analysis_errors": [],  # NUOVO
        "analysis_attempts": 0,  # NUOVO
        "vega_spec": {},
        "plot_html": "",  # Non usato, mantenuto per compatibilità
        "final_response": "",
        "messages": [],
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

        # SALVA SPEC JSON invece di HTML
        if final_state.get("vega_spec"):
            output_file = f"sleep_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_state["vega_spec"], f, indent=2)
            print(f"\n✓ Specifica Vega-Lite salvata in: {output_file}")

    return final_state

def get_chain():
    """Funzione helper per Streamlit"""
    return create_sleep_analysis_chain()


def create_sleep_analysis_chain_with_config_llm(llm_code: ChatGoogleGenerativeAI) -> StateGraph:
    """Versione con LLM configurato"""
    return create_sleep_analysis_chain()


# ==================== MAIN ====================
if __name__ == "__main__":
    queries = [
        "C'è correlazione tra il numero di risvegli e la frequenza cardiaca negli ultimi 10 giorni?"
    ]

    for query in queries:
        result = run_analysis(query)
        print("\n" + "=" * 60 + "\n")
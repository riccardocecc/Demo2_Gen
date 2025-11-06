from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import re
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL

from settings import llm_code
from tool import get_sleep_data

python_repl = PythonREPLTool()
# ==================== STATE DEFINITION ====================
class SleepAnalysisState(TypedDict):
    """State che mantiene lo stato della conversazione attraverso i nodi"""
    query: str
    subject_id: int
    period: str
    raw_data: dict  # Può contenere sleep_data, kitchen_data, o entrambi
    data_sources: list  # ['sleep', 'kitchen'] o ['sleep'] o ['kitchen']
    statistical_method: dict
    analysis_code: str
    analysis_results: dict
    plot_code: str
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
    Utilizza l'LLM con il tool get_sleep_data per estrarre e recuperare i dati.
    """
    print("\n[NODE 1] Estrazione dati dal CSV...")

    query = state["query"]

    # Crea un LLM con binding al tool
    llm_with_tools = llm.bind_tools([get_sleep_data])

    # Prompt per l'LLM che deve chiamare il tool
    extraction_prompt = f"""
    Sei un assistente che deve recuperare dati del sonno usando il tool get_sleep_data, devi SOLO ESTRARRE I PARAMETRI NON CONSIDERARE TUTTO IL RESTO.

    Query dell'utente: "{query}"

    Analizza la query ed estrai:
    1. subject_id: ID del soggetto (numero intero). Se non specificato, usa 1 come default.
    2. period: periodo da analizzare in formato 'last_N_days' (es: 'last_30_days') oppure 'YYYY-MM-DD,YYYY-MM-DD'
       Se non specificato, usa 'last_30_days' come default.

    Chiama il tool get_sleep_data con i parametri appropriati per recuperare i dati.

    Esempi:
    - "analizza il sonno del soggetto 2 negli ultimi 7 giorni" → subject_id=2, period='last_7_days'
    - "dati dal 2024-01-01 al 2024-01-31" → subject_id=1, period='2024-01-01,2024-01-31'
    - "mostra il sonno" → subject_id=1, period='last_30_days'
    """

    try:
        # Invoca l'LLM che chiamerà il tool
        response = llm_with_tools.invoke([HumanMessage(content=extraction_prompt)])

        # Verifica se l'LLM ha chiamato il tool
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]

            # Estrai i parametri
            subject_id = tool_call['args'].get('subject_id', 1)
            period = tool_call['args'].get('period', 'last_30_days')

            print(f" soggetto ID: {subject_id}, period: {period}")

            # Chiama il tool effettivamente
            result = get_sleep_data.invoke({
                'subject_id': subject_id,
                'period': period
            })

            # Verifica se c'è un errore
            if isinstance(result, dict) and 'error' in result:
                state["error"] = result['error']
                return state

            # Salva i dati nello state
            state["subject_id"] = subject_id
            state["period"] = period
            state["raw_data"] = result

            print(f"✓ Estratti {len(result['records'])} record per soggetto {subject_id}")

        else:
            # Fallback: estrai parametri manualmente usando regex
            print("⚠️ LLM non ha chiamato il tool, uso fallback con regex...")

            subject_id = 1  # default
            period = "last_30_days"  # default

            # Cerca subject_id nella query
            subject_match = re.search(r'soggetto[:\s]+(\d+)|subject[:\s]+(\d+)|id[:\s]+(\d+)', query, re.IGNORECASE)
            if subject_match:
                subject_id = int([g for g in subject_match.groups() if g][0])

            # Cerca period nella query
            if 'ultimi' in query.lower() or 'last' in query.lower():
                days_match = re.search(r'(\d+)\s*giorni|(\d+)\s*days', query, re.IGNORECASE)
                if days_match:
                    days = int([g for g in days_match.groups() if g][0])
                    period = f"last_{days}_days"
            elif re.search(r'\d{4}-\d{2}-\d{2}', query):
                dates = re.findall(r'\d{4}-\d{2}-\d{2}', query)
                if len(dates) >= 2:
                    period = f"{dates[0]},{dates[1]}"

            # Chiama il tool con i parametri estratti
            result = get_sleep_data.invoke({
                'subject_id': subject_id,
                'period': period
            })

            if isinstance(result, dict) and 'error' in result:
                state["error"] = result['error']
                return state

            state["subject_id"] = subject_id
            state["period"] = period
            state["raw_data"] = result

            print(f"✓ Estratti {len(result['records'])} record per soggetto {subject_id}")

    except Exception as e:
        state["error"] = f"Errore nell'estrazione dati: {str(e)}"
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    return state


# ==================== NODE 1.5: STATISTICAL METHOD SELECTION ====================
def select_statistical_method_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 1.5: Analizza la query dell'utente e determina quale analisi statistica
    è più appropriata basandosi sui dati disponibili.
    Questo nodo ha LIBERTÀ COMPLETA nella scelta del metodo statistico.
    """
    print("\n[NODE 1.5] Selezione metodo statistico...")

    if state.get("error"):
        return state

    query = state["query"]
    raw_data = state["raw_data"]

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
        # Invoca LLM per selezione metodo
        response = llm.invoke([HumanMessage(content=selection_prompt)])

        # Estrai JSON dalla risposta
        response_text = response.content.strip()

        # Rimuovi eventuali markdown code blocks
        response_text = re.sub(r'```json\n?', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)

        # Parsing JSON
        method_selection = json.loads(response_text)

        # Salva nello state
        state["statistical_method"] = method_selection

        print(f"✓ Obiettivo analisi: {method_selection['analysis_goal']}")
        print(f"  Tipo analisi: {method_selection['analysis_type']}")
        print(f"  Variabili: {method_selection['variables']}")
        print(f"  Metodi statistici: {method_selection['statistical_methods']}")
        print(f"  Visualizzazione: {method_selection['visualization_type']}")

        if method_selection.get('considerations'):
            print(f"Considerazioni: {method_selection['considerations']}")

    except json.JSONDecodeError as e:
        print(f"⚠️ Errore nel parsing JSON, uso fallback...")
        # Fallback: analisi basata su keyword semplici
        query_lower = query.lower()

        # Analisi semplice basata su keywords
        if any(kw in query_lower for kw in ["correlazione", "relazione", "rapporto", "varia rispetto"]):
            analysis_type = "correlation"
            viz_type = "scatter con regressione"
        elif any(kw in query_lower for kw in ["proporzione", "percentuale", "distribuzione", "composizione"]):
            analysis_type = "proportion"
            viz_type = "pie chart o bar chart"
        elif any(kw in query_lower for kw in ["trend", "andamento", "evoluzione", "nel tempo"]):
            analysis_type = "trend"
            viz_type = "line chart"
        elif any(kw in query_lower for kw in ["confronto", "differenza", "vs"]):
            analysis_type = "comparison"
            viz_type = "bar chart comparativo"
        else:
            analysis_type = "descriptive"
            viz_type = "statistiche descrittive"

        method_selection = {
            "analysis_goal": "Analisi automatica basata su keywords",
            "analysis_type": analysis_type,
            "variables": [],
            "statistical_methods": ["auto-detected"],
            "calculations_needed": {},
            "expected_outputs": [],
            "visualization_type": viz_type,
            "reasoning": "Determinato tramite keyword matching (fallback)",
            "considerations": "Analisi automatica - verificare appropriatezza"
        }

        state["statistical_method"] = method_selection
        print(f"✓ Metodo selezionato (fallback): {method_selection['analysis_type']}")

    except Exception as e:
        state["error"] = f"Errore nella selezione del metodo statistico: {str(e)}"
        print(f"❌ Errore: {e}")

    return state


# ==================== NODE 2: STATISTICAL ANALYSIS ====================
def statistical_analysis_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 2: Genera ed esegue codice Python per analisi statistica
    basata RIGOROSAMENTE sul metodo statistico selezionato nel nodo precedente.
    """
    print("\n[NODE 2] Analisi statistica...")

    if state.get("error"):
        return state

    query = state["query"]
    raw_data = state["raw_data"]
    method_selection = state.get("statistical_method", {})

    # Crea prompt per generare codice di analisi
    analysis_prompt = f"""
Sei un esperto di analisi statistica dei dati del sonno.

Query originale: "{query}"

METODO STATISTICO SELEZIONATO (da rispettare RIGOROSAMENTE):
- Obiettivo: {method_selection.get('analysis_goal', '')}
- Tipo di analisi: {method_selection.get('analysis_type', '')}
- Variabili da analizzare: {method_selection.get('variables', [])}
- Metodi statistici: {method_selection.get('statistical_methods', [])}
- Calcoli richiesti: {json.dumps(method_selection.get('calculations_needed', dict()), indent=2)}
- Output attesi: {method_selection.get('expected_outputs', [])}
- Considerazioni: {method_selection.get('considerations', '')}

Dati disponibili (campi nel DataFrame):
- data: data della registrazione (datetime)
- total_sleep_time: durata totale del sonno in minuti
- rem_sleep_duration: durata fase REM in minuti
- deep_sleep_duration: durata sonno profondo in minuti
- light_sleep_duration: durata sonno leggero in minuti
- wakeup_count: numero di risvegli per notte
- out_of_bed_count: numero di uscite dal letto
- hr_average: frequenza cardiaca media (bpm)
- rr_average: frequenza respiratoria media (respiri/min)
- subject_id: ID del soggetto

Numero di record: {len(raw_data['records'])}

ISTRUZIONI CRITICHE:

1. Implementa ESATTAMENTE l'analisi specificata nel metodo statistico selezionato
2. Usa SOLO le variabili indicate in 'variables'
3. Calcola TUTTI i valori richiesti in 'calculations_needed' e 'expected_outputs'
4. Il dizionario 'results' deve contenere TUTTE le metriche specificate

REQUISITI DEL CODICE:
- Assumi che i dati siano in 'df' (DataFrame pandas)
- Importa solo librerie necessarie: numpy, scipy.stats
- NON stampare nulla TRANNE l'ultima riga che deve essere: print(json.dumps(results, default=str))
- Salva TUTTI i risultati nel dizionario 'results'
- Gestisci NaN e valori mancanti
- Converti date in string se necessario per serializzazione
- Assicurati che 'results' contenga TUTTE le metriche richieste
- Alla fine del codice, DEVI stampare results serializzato in JSON con: print(json.dumps(results, default=str))

IMPORTANTE: Il codice deve essere COERENTE con il metodo selezionato.
Se il metodo richiede proporzioni, calcola proporzioni.
Se richiede correlazioni, calcola correlazioni.
Se richiede statistiche descrittive, calcolale.

L'ULTIMA RIGA del tuo codice DEVE essere: print(json.dumps(results, default=str))

Rispondi SOLO con il codice Python eseguibile, senza spiegazioni o markdown.
"""

    try:
        # Genera codice con LLM
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis_code = response.content.strip()

        # Rimuovi eventuali markdown code blocks
        analysis_code = re.sub(r'```python\n?', '', analysis_code)
        analysis_code = re.sub(r'```\n?', '', analysis_code)

        state["analysis_code"] = analysis_code

        # Prepara i dati come DataFrame
        df = pd.DataFrame(raw_data['records'])
        df['data'] = pd.to_datetime(df['data'])

        # Prepara il codice completo con setup iniziale
        full_code = f"""
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import json

# Dati
df = pd.DataFrame({raw_data['records']})
df['data'] = pd.to_datetime(df['data'])

# Inizializza results
results = {{}}

# Codice di analisi generato
{analysis_code}

# Assicurati che results sia stampato (se non già presente nel codice generato)
if 'print(json.dumps(results' not in '''{analysis_code}''':
    print(json.dumps(results, default=str))
"""

        # Esegui con Python REPL
        output = python_repl.run(full_code)

        # Parse JSON dall'output
        if output and output.strip():
            # Pulisci l'output da eventuali messaggi extra
            output_lines = output.strip().split('\n')

            # Cerca l'ultima riga che contiene JSON valido
            json_output = None
            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_output = line
                    break

            if json_output:
                results = json.loads(json_output)

                # ===== AGGIUNGI QUESTA PULIZIA =====
                # Pulisci tutte le stringhe da caratteri di controllo
                cleaned_results = {}
                for key, value in results.items():
                    if isinstance(value, str):
                        # Rimuovi \n, \r, \t e sostituisci con spazio singolo
                        cleaned_value = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                        # Rimuovi spazi multipli consecutivi
                        cleaned_value = ' '.join(cleaned_value.split())
                        cleaned_results[key] = cleaned_value
                    else:
                        cleaned_results[key] = value
                # ===================================

                state["analysis_results"] = cleaned_results  # <-- USA cleaned_results

                print(f"✓ Analisi completata: {len(cleaned_results)} metriche calcolate")
                print(f"  Metriche: {list(cleaned_results.keys())}")

                # Mostra alcuni risultati chiave
                for key, value in list(cleaned_results.items())[:5]:
                    if isinstance(value, (int, float)):
                        print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
                    else:
                        print(f"  - {key}: {value}")
            else:
                state["error"] = "Impossibile trovare JSON valido nell'output"
                print(f"❌ Output ricevuto:\n{output}")
        else:
            state["error"] = "Il codice non ha prodotto output"
            print("❌ Nessun output dal codice di analisi")

    except json.JSONDecodeError as e:
        state["error"] = f"Errore nel parsing JSON dell'output: {str(e)}"
        print(f"❌ Errore JSON: {e}")
        print(f"Output ricevuto:\n{output[:500]}...")  # Mostra primi 500 caratteri
        print(f"\nCodice generato:\n{analysis_code}")

    except Exception as e:
        state["error"] = f"Errore nell'esecuzione dell'analisi: {str(e)}"
        print(f"❌ Errore: {e}")
        print(f"Codice generato:\n{analysis_code}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

    return state

# ==================== NODE 3: PLOT GENERATION ====================
def plot_generation_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 3: Genera ed esegue codice Python per creare un grafico Plotly
    con retry logic, validazione del codice e propagazione corretta degli errori.
    """
    print("\n[NODE 3] Generazione grafico Plotly...")

    # Se c'è già un errore nello state, non procedere
    if state.get("error"):
        return state

    # Estrai dati dallo state
    query = state["query"]
    analysis_results = state["analysis_results"]
    raw_data = state["raw_data"]
    method_selection = state.get("statistical_method", {})
    visualization_type = method_selection.get("visualization_type", "grafico generico")

    # Configurazione retry logic
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"\nTentativo {attempt}/{max_attempts}")

        try:
            # ============================================================
            # STEP 1: Genera il prompt con feedback da tentativi precedenti
            # ============================================================
            plot_prompt = create_plot_prompt(
                query=query,
                visualization_type=visualization_type,
                analysis_results=analysis_results,
                previous_error=state.get("last_plot_error"),
                previous_code=state.get("plot_code")
            )

            # ============================================================
            # STEP 2: Genera codice con LLM
            # ============================================================
            response = llm.invoke([HumanMessage(content=plot_prompt)])
            plot_code = response.content.strip()

            # ============================================================
            # STEP 3: Pulisci il codice generato
            # ============================================================
            plot_code = clean_generated_code(plot_code)

            # ============================================================
            # STEP 4: Valida il codice prima dell'esecuzione
            # ============================================================
            validation_error = validate_plotly_code(plot_code)
            if validation_error:
                print(f"  ⚠️  Validazione fallita: {validation_error}")
                state["last_plot_error"] = validation_error
                continue  # Riprova con il prossimo tentativo

            # Salva il codice nello state
            state["plot_code"] = plot_code

            # ============================================================
            # STEP 5: Esegui il codice e cattura eventuali errori
            # ============================================================

            plot_html, execution_error = execute_plot_code(
                plot_code=plot_code,
                raw_data=raw_data,
                analysis_results=analysis_results,
                method_selection=method_selection
            )

            # ============================================================
            # STEP 6: Verifica il risultato
            # ============================================================
            if plot_html:
                # Successo! Rimuovi eventuali errori precedenti
                state["plot_html"] = plot_html
                state.pop("last_plot_error", None)
                print(" Grafico generato con successo!")
                return state
            else:
                # Esecuzione fallita, solleva eccezione con errore dettagliato
                error_msg = execution_error or "HTML del grafico vuoto (causa sconosciuta)"
                raise ValueError(error_msg)

        except Exception as e:
            # ============================================================
            # GESTIONE ERRORI
            # ============================================================
            error_msg = str(e)
            print(f"  Errore tentativo {attempt}: {error_msg[:200]}...")

            # Salva l'errore per il prossimo tentativo
            state["last_plot_error"] = error_msg

            # Se è l'ultimo tentativo, salva l'errore definitivo nello state
            if attempt == max_attempts:
                state["error"] = (
                    f"Impossibile generare il grafico dopo {max_attempts} tentativi.\n\n"
                    f"Ultimo errore: {error_msg}"
                )
                print(f"\n ERRORE FINALE dopo {max_attempts} tentativi")
                print(f"\nCodice generato nell'ultimo tentativo:")
                print("=" * 60)
                print(state.get('plot_code', 'N/A'))
                print("=" * 60)

                # Log traceback completo per debug
                import traceback
                print(f"\nTraceback completo:")
                print(traceback.format_exc())

    # Se usciamo dal loop senza successo, lo state contiene già l'errore
    return state

def create_plot_prompt(query: str, visualization_type: dict, analysis_results: dict,
                       previous_error: str = None, previous_code: str = None) -> str:
    """Crea il prompt per la generazione del grafico con feedback iterativo"""

    base_prompt = f"""
Sei un esperto nella visualizzazione dati con Plotly per analisi del sonno.

Query originale: "{query}"

RISULTATI DELL'ANALISI STATISTICA (da integrare nel grafico):
{json.dumps(analysis_results, indent=2, default=str)}

COLONNE DISPONIBILI NEL DATAFRAME 'df':
- 'data': data della registrazione (datetime)
- 'total_sleep_time': durata totale del sonno in minuti
- 'rem_sleep_duration': durata fase REM in minuti
- 'deep_sleep_duration': durata sonno profondo in minuti
- 'light_sleep_duration': durata sonno leggero in minuti
- 'wakeup_count': numero di risvegli per notte
- 'out_of_bed_count': numero di uscite dal letto
- 'hr_average': frequenza cardiaca media (bpm)
- 'rr_average': frequenza respiratoria media (respiri/min)
- 'subject_id': ID del soggetto

ISTRUZIONI CRITICHE:

1. Crea un grafico: {visualization_type}

3. Usa ESATTAMENTE le variabili da method_selection['variables']

4. NON inventare nomi di colonne che non esistono

REQUISITI TECNICI:
- Crea la figura e salvala nella variabile 'fig'
- Usa titoli descrittivi
- Aggiungi label agli assi
- Usa layout professionale
- Gestisci NaN nei dati con dropna() se necessario
- NON stampare nulla, NON usare fig.show()
- USA SOLO queste librerie già importate: pandas (pd), numpy (np), plotly.graph_objects (go), plotly.express (px)
- NON IMPORTARE !!! statsmodels, scikit-learn, matplotlib, seaborn o altre librerie esterne


ERRORI DA EVITARE ASSOLUTAMENTE:
1. NON usare proprietà 'current' o 'current_value' in rangeselector buttons
2. USA SOLO proprietà valide per rangeselector.Button:
   - count: numero di step
   - label: etichetta del bottone
   - step: unità (day, month, year, etc.)
   - stepmode: "backward" o "todate"
   - visible: True/False

   Esempio CORRETTO:
   dict(count=1, label="1M", step="month", stepmode="backward")

3. NON usare colonne che non esistono nel DataFrame
4. Gestisci valori NaN prima di plottare
5. Verifica che 'fig' sia una figura Plotly valida

Variabili disponibili:
- df: DataFrame pandas
- results: dict con risultati analisi
- method_selection: dict con metodo selezionato
- go: plotly.graph_objects
- px: plotly.express
- pd: pandas
- np: numpy

Rispondi SOLO con codice Python eseguibile che crea 'fig'.
"""

    # Aggiungi feedback da tentativi precedenti
    if previous_error and previous_code:
        feedback = f"""

ERRORE NEL TENTATIVO PRECEDENTE:
{previous_error}

CODICE CHE HA CAUSATO L'ERRORE:
{previous_code}

CORREGGI L'ERRORE E RIGENERA IL CODICE.
Analizza attentamente l'errore e assicurati di non ripeterlo.
"""
        base_prompt += feedback

    return base_prompt


def clean_generated_code(code: str) -> str:
    """Pulisce il codice generato dall'LLM"""

    # Rimuovi markdown code blocks
    code = re.sub(r'```python\n?', '', code)
    code = re.sub(r'```\n?', '', code)

    # Rimuovi chiamate a show/save
    code = re.sub(r'fig\.show\(\)', '', code)
    code = re.sub(r'fig\.write_html\(.*?\)', '', code)
    code = re.sub(r'fig\.to_html\(.*?\)', '', code)

    # Correzioni automatiche per errori comuni
    code = re.sub(r'["\']?current["\']?\s*:', '"count":', code)
    code = re.sub(r'current_value', 'count', code)
    code = re.sub(r'["\']?curr["\']?\s*:', '"count":', code)

    return code.strip()


def validate_plotly_code(code: str) -> str:
    """
    Valida il codice Plotly generato.
    Ritorna stringa di errore se invalido, None se valido.
    """

    # Check 1: Verifica che 'fig' venga creato
    if 'fig' not in code:
        return "Il codice non crea una variabile 'fig'"

    # Check 2: Verifica proprietà invalide comuni
    invalid_patterns = [
        (r'["\']current["\']', "Proprietà 'current' non valida in rangeselector"),
        (r'current_value', "Proprietà 'current_value' non valida"),
    ]

    for pattern, error_msg in invalid_patterns:
        if re.search(pattern, code):
            return error_msg

    # Check 3: Verifica che non ci siano comandi pericolosi
    dangerous_patterns = [
        r'import\s+os',
        r'import\s+sys',
        r'exec\(',
        r'eval\(',
        r'__import__',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return f"Codice contiene pattern pericoloso: {pattern}"

    return None


def execute_plot_code(plot_code: str, raw_data: dict,
                      analysis_results: dict, method_selection: dict) -> str:
    """Esegue il codice del grafico in modo sicuro e ritorna l'HTML"""

    # Serializza i dati
    results_json = json.dumps(analysis_results, default=str)
    method_selection_json = json.dumps(method_selection, default=str)
    raw_data_json = json.dumps(raw_data['records'], default=str)

    # Crea il codice completo
    full_code = f"""import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Carica dati
raw_data_list = json.loads('''{raw_data_json}''')
results = json.loads('''{results_json}''')
method_selection = json.loads('''{method_selection_json}''')

# Crea DataFrame
df = pd.DataFrame(raw_data_list)
df['data'] = pd.to_datetime(df['data'])

# Codice generato
{plot_code}

# Verifica fig
if 'fig' not in locals():
    raise ValueError("La variabile 'fig' non è stata creata")

# Converti in HTML
html_output = fig.to_html(include_plotlyjs='cdn', full_html=True)
print("__PLOT_START__")
print(html_output)
print("__PLOT_END__")
"""
    try:
        output = python_repl.run(full_code)
        if output:
            lines = output.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('<!DOCTYPE') or line.strip().startswith('<html'):
                    return '\n'.join(lines[i:]), None


        return None, f"Nessun HTML trovato nell'output:\n{output[:500]}"
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return None, error_detail


#====================CORRELATION NODE====================
# ==================== NODE 4: NATURAL LANGUAGE RESPONSE ====================
def generate_response_node(state: SleepAnalysisState, llm: ChatGoogleGenerativeAI) -> SleepAnalysisState:
    """
    Nodo 4: Genera una risposta in linguaggio naturale per medici
    basata sui risultati dell'analisi statistica.
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
Sei un assistente medico che deve spiegare risultati statistici a un medico in modo chiaro e clinicamente rilevante.

QUERY ORIGINALE DEL MEDICO:
"{query}"

CONTESTO DELL'ANALISI:
- Paziente ID: {subject_id}
- Periodo analizzato: {period}
- Numero di notti: {num_records}
- Tipo di analisi: {method_selection.get('analysis_type', 'N/A')}

RISULTATI STATISTICI:
{json.dumps(analysis_results, indent=2, default=str)}

ISTRUZIONI:
1. Rispondi DIRETTAMENTE alla domanda del medico
2. Usa linguaggio clinico ma accessibile (evita gergo statistico complesso), NON FARE SUPPOSIZIONI CLINICHE SUI DEI DATI. LIMITATI A RISPONDERE
3. Struttura la risposta in 3 parti:
   - Risposta diretta (2-3 frasi)

4. Traduci valori statistici in termini comprensibili, esempio:
   - Invece di "p-value < 0.05" → "statisticamente significativo"
   - Invece di "r = 0.98" → "forte correlazione positiva"
   - Converti unità se necessario (minuti → ore, etc.)

5. Sii conciso: massimo 150 parole totali

6. NON usare formattazione markdown, grassetto, o titoli
7. Scrivi in paragrafi continui, senza elenchi puntati

Rispondi SOLO con il testo della risposta, senza preamboli.
"""

    try:
        response = llm.invoke([HumanMessage(content=response_prompt)])
        final_response = response.content.strip()

        state["final_response"] = final_response
        print(f"✓ Risposta generata ({len(final_response)} caratteri)")

    except Exception as e:
        state["error"] = f"Errore nella generazione risposta: {str(e)}"
        print(f"❌ Errore: {e}")

    return state
#==================== GRAPH CREATION ====================
def create_sleep_analysis_chain() -> StateGraph:
    """
    Crea la chain LangGraph con i 4 nodi.
    """

    workflow = StateGraph(SleepAnalysisState)


    workflow.add_node("extract_data", lambda state: extract_sleep_data_node(state, llm_code))
    workflow.add_node("select_method", lambda state: select_statistical_method_node(state, llm_code))
    workflow.add_node("analyze", lambda state: statistical_analysis_node(state, llm_code))
    workflow.add_node("plot", lambda state: plot_generation_node(state, llm_code))
    workflow.add_node("respond", lambda state: generate_response_node(state, llm_code))  # <-- NUOVO


    workflow.set_entry_point("extract_data")

    # Funzione per controllare errori dopo extract_data
    def check_error_extract(state: SleepAnalysisState) -> Literal["select_method", "end"]:
        if state.get("error"):
            return "end"
        return "select_method"

    # Funzione per controllare errori dopo select_method
    def check_error_method(state: SleepAnalysisState) -> Literal["analyze", "end"]:
        if state.get("error"):
            return "end"
        return "analyze"

    # Funzione per controllare errori dopo analyze
    def check_error_analyze(state: SleepAnalysisState) -> Literal["plot", "end"]:
        if state.get("error"):
            return "end"
        return "plot"

    def check_error_plot(state: SleepAnalysisState) -> Literal["respond", "end"]:
        if state.get("error"):
            return "end"
        return "respond"


    # Edge condizionale dopo extract_data → select_method
    workflow.add_conditional_edges(
        "extract_data",
        check_error_extract,
        {
            "select_method": "select_method",
            "end": END
        }
    )

    # Edge condizionale dopo select_method → analyze
    workflow.add_conditional_edges(
        "select_method",
        check_error_method,
        {
            "analyze": "analyze",
            "end": END
        }
    )

    # Edge condizionale dopo analyze → plot
    workflow.add_conditional_edges(
        "analyze",
        check_error_analyze,
        {
            "plot": "plot",
            "end": END
        }
    )

    workflow.add_conditional_edges("plot", check_error_plot,
                                   {"respond": "respond", "end": END})

    # Edge finale: plot → END
    workflow.add_edge("respond", END)
    #graph = workflow.compile()
    #png_data = graph.get_graph().draw_mermaid_png()

    #output_file = "sleep_analysis_graph.png"
    #with open(output_file, "wb") as f:
    #    f.write(png_data)
    #print(f"✓ Grafo salvato in: {output_file}")
    return workflow.compile()


# ==================== UTILITY FUNCTIONS ====================
def run_analysis(query: str):
    """
    Esegue l'intera pipeline di analisi.
    """
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
        "plot_code": "",
        "plot_html": "",
        "final_response": "",  # <-- NUOVO
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
        print(f"✓ Records analizzati: {len(final_state['raw_data']['records'])}")

        # Mostra la risposta clinica
        print(f"\n{'=' * 60}")
        print("RISPOSTA CLINICA")
        print(f"{'=' * 60}")
        print(final_state['final_response'])
        print(f"{'=' * 60}")

        # Dettagli tecnici (opzionale, per debug)
        if final_state.get("plot_html"):
            output_file = f"sleep_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_state["plot_html"])
            print(f"\n✓ Grafico salvato in: {output_file}")

    return final_state

def create_sleep_analysis_chain_with_config_llm(llm_code: ChatGoogleGenerativeAI) -> StateGraph:
    """
    Versione per usare con llm_code già configurato da backend.config.settings
    """
    return create_sleep_analysis_chain()

# Alla fine del file, aggiungi:
def get_chain():
    """Funzione helper per Streamlit"""
    return create_sleep_analysis_chain()

# ==================== MAIN ====================
if __name__ == "__main__":

    queries = [
        "C'è correlazione tra il numero di risvegli e la frequenza cardiaca negli ultimi 10 giorni?"
    ]

    for query in queries:
        result = run_analysis(query)
        print("\n" + "=" * 60 + "\n")
from datetime import timedelta
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel

from settings import SLEEP_DATA_PATH
from type import SleepDataResult, ErrorResult, KitchenDataResult

class ToolInput(BaseModel):
    subject_id: Annotated[int, "ID of the subject to retrieve data for, integer"]
    period: Annotated[str, "Period to retrieve in format 'YYYY-MM-DD,YYYY-MM-DD' or 'last_N_days' (e.g., 'last_30_days')"]

@tool(args_schema=ToolInput)
def get_sleep_data(
        subject_id: int,
        period: str
) -> SleepDataResult | ErrorResult:
    """
    Recupera i dati grezzi del sonno per un soggetto specifico in un periodo definito.

    Questo tool restituisce tutti i record del CSV filtrati per soggetto e periodo,
    senza effettuare calcoli o aggregazioni. Ogni record contiene:
    - data: data della registrazione
    - total_sleep_time: durata totale del sonno in minuti
    - rem_sleep_duration: durata fase REM in minuti
    - deep_sleep_duration: durata sonno profondo in minuti
    - light_sleep_duration: durata sonno leggero in minuti
    - wakeup_count: numero di risvegli per notte
    - out_of_bed_count: numero di uscite dal letto per notte
    - hr_average: frequenza cardiaca media notturna (bpm)
    - rr_average: frequenza respiratoria media (respiri/min)

    Usa questo tool quando l'utente chiede:
    - "Mostrami i dati del sonno"
    - "Dati grezzi del sonno"
    - "Record del sonno per periodo"
    - "Visualizza le registrazioni del sonno"

    Args:
        subject_id: ID numerico del soggetto
        period: Periodo in formato 'last_N_days' o 'YYYY-MM-DD,YYYY-MM-DD'

    Returns:
        SleepDataResult con lista di record filtrati, oppure ErrorResult
    """
    try:
        df = pd.read_csv(SLEEP_DATA_PATH)
        df['data'] = pd.to_datetime(df['data'])

        # Filtra per soggetto
        df_subject = df[df['subject_id'] == subject_id].copy()

        if df_subject.empty:
            return ErrorResult(error=f"Nessun dato trovato per il soggetto {subject_id}")

        # Parsing del periodo
        if period.startswith('last_'):
            days = int(period.split('_')[1])
            end_date = df_subject['data'].max()
            start_date = end_date - timedelta(days=days)
        else:
            dates = period.split(',')
            start_date = pd.to_datetime(dates[0])
            end_date = pd.to_datetime(dates[1])

        # Filtra per periodo
        df_period = df_subject[(df_subject['data'] >= start_date) &
                               (df_subject['data'] <= end_date)]

        if df_period.empty:
            return ErrorResult(error="Nessun dato disponibile per il periodo specificato")

        # Converti i dati in lista di dizionari
        records = df_period.to_dict('records')

        # Converti le date in stringhe per la serializzazione
        for record in records:
            record['data'] = record['data'].strftime('%Y-%m-%d')

        result: SleepDataResult = {
            "subject_id": subject_id,
            "period": f"{start_date.date()} to {end_date.date()}",
            "records": records
        }

        return result

    except Exception as e:
        return ErrorResult(error=f"Errore nel recupero dei dati: {str(e)}")



from datetime import timedelta
from typing import Annotated, TypedDict

import pandas as pd
from langchain_core.tools import tool

from settings import KITCHEN_DATA_PATH  # Assicurati di aggiungere questo in settings.py



@tool(args_schema=ToolInput)
def get_kitchen_data(
        subject_id: int,
        period: str
) -> KitchenDataResult | ErrorResult:
    """
    Recupera i dati grezzi delle attività in cucina per un soggetto specifico in un periodo definito.

    Questo tool restituisce tutti i record del CSV filtrati per soggetto e periodo,
    senza effettuare calcoli o aggregazioni. Ogni record contiene:
    - timestamp_picco: momento di massima temperatura durante l'attività
    - temperatura_max: temperatura massima raggiunta (°C)
    - id_attivita: identificatore univoco dell'attività
    - start_time_attivita: inizio dell'attività in cucina
    - end_time_attivita: fine dell'attività in cucina
    - durata_attivita_minuti: durata dell'attività in minuti
    - fascia_oraria: fascia del giorno (mattina, pomeriggio, sera, notte)
    - subject_id: ID del soggetto

    Usa questo tool quando l'utente chiede:
    - "Mostrami le attività in cucina"
    - "Dati della cucina"
    - "Quando ha usato la cucina"
    - "Attività culinarie del paziente"
    - "Frequenza uso cucina"

    Args:
        subject_id: ID numerico del soggetto
        period: Periodo in formato 'last_N_days' o 'YYYY-MM-DD,YYYY-MM-DD'

    Returns:
        KitchenDataResult con lista di record filtrati, oppure ErrorResult
    """
    try:
        df = pd.read_csv(KITCHEN_DATA_PATH)
        df['timestamp_picco'] = pd.to_datetime(df['timestamp_picco'])
        df['start_time_attivita'] = pd.to_datetime(df['start_time_attivita'])
        df['end_time_attivita'] = pd.to_datetime(df['end_time_attivita'])

        # Filtra per soggetto
        df_subject = df[df['subject_id'] == subject_id].copy()

        if df_subject.empty:
            return ErrorResult(error=f"Nessun dato trovato per il soggetto {subject_id}")

        # Parsing del periodo
        if period.startswith('last_'):
            days = int(period.split('_')[1])
            end_date = df_subject['timestamp_picco'].max()
            start_date = end_date - timedelta(days=days)
        else:
            dates = period.split(',')
            start_date = pd.to_datetime(dates[0])
            end_date = pd.to_datetime(dates[1])

        # Filtra per periodo usando timestamp_picco come riferimento temporale
        df_period = df_subject[(df_subject['timestamp_picco'] >= start_date) &
                               (df_subject['timestamp_picco'] <= end_date)]

        if df_period.empty:
            return ErrorResult(error="Nessun dato disponibile per il periodo specificato")

        # Converti i dati in lista di dizionari
        records = df_period.to_dict('records')

        # Converti le date in stringhe per la serializzazione
        for record in records:
            record['timestamp_picco'] = record['timestamp_picco'].strftime('%Y-%m-%d %H:%M:%S')
            record['start_time_attivita'] = record['start_time_attivita'].strftime('%Y-%m-%d %H:%M:%S')
            record['end_time_attivita'] = record['end_time_attivita'].strftime('%Y-%m-%d %H:%M:%S')

        result: KitchenDataResult = {
            "subject_id": subject_id,
            "period": f"{start_date.date()} to {end_date.date()}",
            "num_records": len(records),
            "records": records
        }

        return result

    except Exception as e:
        return ErrorResult(error=f"Errore nel recupero dei dati: {str(e)}")
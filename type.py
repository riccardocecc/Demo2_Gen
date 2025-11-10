from typing import TypedDict


class SleepRecord(TypedDict):
    """Singolo record di dati del sonno"""
    data: str
    subject_id: int
    total_sleep_time: float
    rem_sleep_duration: float
    deep_sleep_duration: float
    light_sleep_duration: float
    wakeup_count: int
    out_of_bed_count: int
    hr_average: float
    rr_average: float


class SleepDataResult(TypedDict):
    """Risultato della query per i dati del sonno"""
    subject_id: int
    period: str
    num_records: int
    records: list[SleepRecord]

class ErrorResult(TypedDict):
    error: str

class KitchenRecord(TypedDict):
    """Singolo record di attività in cucina"""
    timestamp_picco: str
    temperatura_max: float
    id_attivita: int
    start_time_attivita: str
    end_time_attivita: str
    durata_attivita_minuti: int
    fascia_oraria: str
    subject_id: int


class KitchenDataResult(TypedDict):
    """Risultato della query per i dati della cucina"""
    subject_id: int
    period: str
    num_records: int
    records: list[KitchenRecord]


class ErrorResult(TypedDict):
    error: str

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
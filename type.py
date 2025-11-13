from typing import TypedDict, List, Literal, Annotated
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
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
    plotly_figure_dict:dict
    plot_attempts: int
    plot_errors: list
    code_response: str

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
    expected_outputs: List[str] = Field(
        description="Lista degli output attesi dall'analisi"
    )
    visualization_type: str = Field(
        description="Se necessario, descrivi brevemente che tipo di visualizzazione plotly è necessaria, NO GRAPH altrimenti"
    )
    calculations_needed: dict = Field(
        description="Dizionario con nome metrica e descrizione del calcolo"
    )

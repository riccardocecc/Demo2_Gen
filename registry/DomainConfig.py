from typing import Dict, Any
from abc import ABC, abstractmethod
from typing import TypedDict, Literal, Dict, Any, List, Optional, Protocol
import pandas as pd


class DataExtractor:
    def extract(self, state: Dict[str, Any]) -> Dict[str, Any]: ...


class DomainConfig(ABC):
    """Classe base astratta per la configurazione del dominio"""

    @abstractmethod
    def get_available_columns(self) -> List[str]:
        """Restituisce la lista di colonne disponibili per questo dominio"""
        pass

    @abstractmethod
    def get_columns_with_types(self) -> Dict[str, str]:
        """Restituisce un dizionario con le colonne e i loro tipi di dato"""
        pass

    @abstractmethod
    def get_domain_name(self) -> str:
        """Restituisce il nome del dominio"""
        pass

    @abstractmethod
    def get_dataframe_name(self) -> str:
        """Restituisce il nome del dataframe associato a questo dominio"""
        pass
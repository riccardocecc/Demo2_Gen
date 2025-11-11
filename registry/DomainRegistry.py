from typing import Dict, Optional, List, Any, Tuple

from datacleaner import DataCleaner
from domain_configs import SLEEP_CONFIG
from registry.DomainConfig import DomainConfig, DataExtractor
import pandas as pd


class DomainRegistry:
    """Registry per gestire diversi domini"""

    def __init__(self):
        self._domains: Dict[str, DomainConfig] = {}

    def register(self, name: str):
        def decorator(domain_class):
            self._domains[name] = domain_class()
            return domain_class

        return decorator

    def get_domain(self, name: str) -> Optional[DomainConfig]:
        return self._domains.get(name)

    def list_domains(self) -> List[str]:
        return list(self._domains.keys())

    def get_available_columns_for_domains(self, domain_names: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Restituisce un dizionario con le colonne disponibili e i loro tipi per ogni dataframe.

        Returns:
            Dict[str, Dict[str, str]]: {dataframe_name: {col_name: data_type, ...}}

        Example:
            {
                'get_sleep_data': {
                    'data': 'datetime',
                    'total_sleep_time': 'float',
                    'wakeup_count': 'int',
                    ...
                },
                'get_kitchen_data': {
                    'timestamp_picco': 'datetime',
                    'id_attivita': 'int',
                    ...
                }
            }
        """
        columns_by_dataframe = {}

        for domain_name in domain_names:
            domain = self.get_domain(domain_name)
            if domain:
                dataframe_name = domain.get_dataframe_name()
                columns_by_dataframe[dataframe_name] = domain.get_columns_with_types()

        return columns_by_dataframe


# Crea l'istanza globale del registry
domain_registry = DomainRegistry()


class SleepDataExtractor:
    pass


@domain_registry.register('sleep')
class SleepDomainConfig(DomainConfig):

    def get_available_columns(self) -> List[str]:
        return [
            'data', 'total_sleep_time', 'rem_sleep_duration',
            'deep_sleep_duration', 'light_sleep_duration', 'wakeup_count',
            'out_of_bed_count', 'hr_average', 'rr_average', 'subject_id'
        ]

    def get_columns_with_types(self) -> Dict[str, str]:
        return {
            'data': 'str',  # Solo data, non datetime
            'total_sleep_time': 'float',
            'rem_sleep_duration': 'float',
            'deep_sleep_duration': 'float',
            'light_sleep_duration': 'float',
            'wakeup_count': 'int',
            'out_of_bed_count': 'int',
            'hr_average': 'float',
            'rr_average': 'float',
            'subject_id': 'int'
        }

    def get_domain_name(self) -> str:
        return "sleep"

    def get_dataframe_name(self) -> str:
        return "get_sleep_data"


@domain_registry.register('kitchen')
class KitchenDomainConfig(DomainConfig):
    """Configurazione dominio Cucina"""

    def get_available_columns(self) -> List[str]:
        return [
            'timestamp_picco', 'temperatura_max', 'id_attivita',
            'start_time_attivita', 'end_time_attivita', 'durata_attivita_minuti',
            'fascia_oraria', 'subject_id'
        ]

    def get_columns_with_types(self) -> Dict[str, str]:
        return {
            'timestamp_picco': 'str',  # Convertito in solo data per merge con sleep
            'temperatura_max': 'float',
            'id_attivita': 'int',
            'start_time_attivita': 'time',  # Solo ora
            'end_time_attivita': 'time',  # Solo ora
            'durata_attivita_minuti': 'float',
            'fascia_oraria': 'string',
            'subject_id': 'int'
        }

    def get_domain_name(self) -> str:
        return "kitchen"

    def get_dataframe_name(self) -> str:
        return "get_kitchen_data"
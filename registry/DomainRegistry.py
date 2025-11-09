from typing import Dict, Optional, List, Any

from datacleaner import DataCleaner
from domain_configs import SLEEP_CONFIG
from registry.DomainConfig import DomainConfig, DataExtractor
import pandas as pd

class DomainRegistry:
    """Registry per gestire diversi domini"""

    def __init__(self):
        # CORREZIONE: separa il type hint dall'inizializzazione
        self._domains: Dict[str, DomainConfig] = {}  # <-- Così è corretto

    def register(self, name: str):  # <-- Aggiungi anche qui il type hint
        def decorator(domain_class):
            self._domains[name] = domain_class()
            return domain_class
        return decorator

    def get_domain(self, name: str) -> Optional[DomainConfig]:
        return self._domains.get(name)

    def list_domains(self) -> List[str]:
        return list(self._domains.keys())

    def get_available_columns_for_domains(self, domain_names: List[str]) -> List[str]:
        columns = []
        for domain_name in domain_names:
            domain = self.get_domain(domain_name)
            if domain:
                columns.extend(domain.get_available_columns())

        return list(set(columns))

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

    def get_domain_name(self) -> str:
        return "sleep"


@domain_registry.register('kitchen')
class KitchenDomainConfig(DomainConfig):
    """Configurazione dominio Cucina"""

    def get_available_columns(self) -> List[str]:
        return [
            'timestamp_picco', 'temperatura_max', 'id_attivita',
            'start_time_attivita', 'end_time_attivita', 'durata_attivita_minuti',
            'fascia_oraria', 'subject_id'
        ]

    def get_domain_name(self) -> str:
        return "kitchen"
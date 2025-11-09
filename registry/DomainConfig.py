from typing import Dict, Any
from abc import ABC, abstractmethod
from typing import TypedDict, Literal, Dict, Any, List, Optional, Protocol
import pandas as pd

class DataExtractor:
    def extract(self, state:Dict[str,Any]) -> Dict[str,Any]: ...

class DomainConfig(ABC):
    """classe base astratta per la configurazione del domino"""


    @abstractmethod
    def get_available_columns(self) -> List[str]:
        pass

    @abstractmethod
    def get_domain_name(self) -> str:
        pass


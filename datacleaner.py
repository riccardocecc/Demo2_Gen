# data_cleaner.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class DomainConfig:
    """Configurazione per la pulizia di un dominio specifico"""
    name: str
    date_columns: List[str]  # Colonne da convertire in datetime
    primary_date_column: str  # Colonna principale per ordinamento/deduplicazione
    numeric_exclude: List[str] = None  # Colonne numeriche da escludere dalla pulizia
    outlier_columns: List[str] = None  # Colonne per rilevamento outliers
    duplicate_subset: List[str] = None  # Colonne per rilevare duplicati
    outlier_iqr_multiplier: float = 3.0  # Moltiplicatore IQR per outliers


class DataCleaner:
    """Classe per la pulizia standardizzata dei dati di diversi domini"""

    def __init__(self, config: DomainConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.cleaning_stats = {}

    def clean(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Esegue la pulizia completa dei dati seguendo la pipeline standard

        Args:
            records: Lista di record dal tool

        Returns:
            DataFrame pulito
        """
        df = pd.DataFrame(records)
        initial_rows = len(df)

        if self.verbose:
            print(f"✓ Dati estratti: {initial_rows} record")
            print(f"  Pulizia dati in corso...")

        # Pipeline di pulizia
        df = self._convert_dates(df)
        df = self._clean_numeric_values(df)
        df = self._remove_invalid_dates(df)
        df = self._remove_duplicates(df)
        df = self._sort_data(df)
        df = self._handle_outliers(df)

        if self.verbose:
            print(f"✓ Pulizia completata: {len(df)} record validi")
            self._print_summary(initial_rows, len(df))

        return df

    def _convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte le colonne date in formato datetime"""
        for col in self.config.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def _clean_numeric_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pulisce i valori numerici (negativi, infiniti)"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Escludi colonne specifiche
        exclude_cols = self.config.numeric_exclude or []
        exclude_cols.append('subject_id')  # Sempre escluso

        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        for col in numeric_cols:
            # Rimuovi valori negativi e infiniti
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        return df

    def _remove_invalid_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rimuove righe con date mancanti nella colonna principale"""
        initial_rows = len(df)
        df = df.dropna(subset=[self.config.primary_date_column])

        removed = initial_rows - len(df)
        if removed > 0 and self.verbose:
            print(f"  Rimosse {removed} righe con date mancanti")

        self.cleaning_stats['invalid_dates'] = removed
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rimuove duplicati basandosi sul subset specificato"""
        initial_rows = len(df)

        subset = self.config.duplicate_subset or [self.config.primary_date_column, 'subject_id']
        df = df.drop_duplicates(subset=subset, keep='first')

        removed = initial_rows - len(df)
        if removed > 0 and self.verbose:
            print(f"  Rimosse {removed} righe duplicate")

        self.cleaning_stats['duplicates'] = removed
        return df

    def _sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordina i dati per la colonna data principale"""
        return df.sort_values(self.config.primary_date_column).reset_index(drop=True)

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rileva e gestisce outliers usando il metodo IQR"""
        if not self.config.outlier_columns:
            return df

        for col in self.config.outlier_columns:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            multiplier = self.config.outlier_iqr_multiplier
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outliers > 0:
                if self.verbose:
                    print(f"  Rilevati {outliers} outliers in '{col}' (impostati a NaN)")
                df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan

                if col not in self.cleaning_stats:
                    self.cleaning_stats[col + '_outliers'] = 0
                self.cleaning_stats[col + '_outliers'] += outliers

        return df

    def _print_summary(self, initial: int, final: int):
        """Stampa un riepilogo della pulizia"""
        if self.cleaning_stats:
            print(f"\n  Riepilogo pulizia '{self.config.name}':")
            for key, value in self.cleaning_stats.items():
                print(f"    - {key}: {value}")
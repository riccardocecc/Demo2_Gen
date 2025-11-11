# domain_configs.py
from datacleaner import DomainConfig


SLEEP_CONFIG = DomainConfig(
    name="sleep",
    date_columns=["data"],
    primary_date_column="data",
    date_only_columns=["data"],  # Solo data, no timestamp
    outlier_columns=["total_sleep_time", "hr_average", "rr_average"]
)

# Configurazione per kitchen data
KITCHEN_CONFIG = DomainConfig(
    name="Cucina",
    date_columns=['timestamp_picco', 'start_time_attivita', 'end_time_attivita'],
    primary_date_column='timestamp_picco',
    numeric_exclude=['id_attivita', 'subject_id'],
    outlier_columns=['temperatura_max', 'durata_attivita_minuti'],
    duplicate_subset=['timestamp_picco', 'subject_id'],
    outlier_iqr_multiplier=3.0,
    date_only_columns=['timestamp_picco'],
    time_columns=['start_time_attivita', 'end_time_attivita']
)


ACTIVITY_CONFIG = DomainConfig(
    name="Attività Fisica",
    date_columns=['data', 'timestamp'],
    primary_date_column='data',
    numeric_exclude=['step_goal'],  # Non pulire il goal giornaliero
    outlier_columns=['steps', 'calories', 'distance_km'],
    duplicate_subset=['data', 'subject_id'],
    outlier_iqr_multiplier=2.5  # Più stringente
)


DOMAIN_CONFIGS = {
    'sleep': SLEEP_CONFIG,
    'kitchen': KITCHEN_CONFIG,
    'activity': ACTIVITY_CONFIG,
}


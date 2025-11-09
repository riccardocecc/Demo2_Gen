# domain_configs.py
from datacleaner import DomainConfig


SLEEP_CONFIG = DomainConfig(
    name="Sonno",
    date_columns=['data'],
    primary_date_column='data',
    numeric_exclude=[],
    outlier_columns=['total_sleep_time', 'hr_average', 'rr_average', 'wakeup_count'],
    duplicate_subset=['data', 'subject_id'],
    outlier_iqr_multiplier=3.0
)


KITCHEN_CONFIG = DomainConfig(
    name="Cucina",
    date_columns=['timestamp_picco', 'start_time_attivita', 'end_time_attivita'],
    primary_date_column='timestamp_picco',
    numeric_exclude=[],
    outlier_columns=['temperatura_max', 'durata_attivita_minuti'],
    duplicate_subset=['id_attivita', 'subject_id'],  # id_attivita è univoco
    outlier_iqr_multiplier=3.0
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


"""
Modulo per la generazione di dati fittizi per la demo del sistema multi-agente.
I dati simulano 180 giorni di raccolta per 3 soggetti.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_sleep_data():

    np.random.seed(42)
    data = []
    start_date = datetime(2024, 1, 1)

    for subject in [1, 2, 3]:
        for day in range(180):
            date = start_date + timedelta(days=day)
            trend_factor = 1.0 + (day / 180) * 0.2 * np.random.normal(0, 1)

            data.append({
                'data': date.strftime('%Y-%m-%d'),
                'total_sleep_time': max(300, 420 + np.random.normal(0, 60) * trend_factor),
                'rem_sleep_duration': max(30, 90 + np.random.normal(0, 20) * trend_factor),
                'deep_sleep_duration': max(45, 120 + np.random.normal(0, 30) * trend_factor),
                'light_sleep_duration': max(100, 210 + np.random.normal(0, 40) * trend_factor),
                'wakeup_count': max(1, int(3 + np.random.poisson(2) * trend_factor)),
                'out_of_bed_count': max(0, int(1 + np.random.poisson(1) * trend_factor)),
                'hr_average': 60 + np.random.normal(0, 8),
                'rr_average': 16 + np.random.normal(0, 3),
                'subject_id': subject
            })

    return pd.DataFrame(data)


def generate_kitchen_data():
    """
    Genera dati fittizi per l'attività in cucina.
    Simula attività di cottura in diverse fasce orarie.
    """
    np.random.seed(43)
    data = []
    start_date = datetime(2024, 1, 1)

    for subject in [1, 2, 3]:
        for day in range(180):
            date = start_date + timedelta(days=day)
            n_activities = np.random.poisson(3)

            for _ in range(n_activities):
                start_time = date + timedelta(
                    hours=np.random.choice([7, 12, 19]) + np.random.normal(0, 1)
                )
                duration = max(5, int(np.random.exponential(20)))

                data.append({
                    'timestamp_picco': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'temperatura_max': 25 + np.random.exponential(15),
                    'id_attivita': len(data) + 1,
                    'start_time_attivita': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time_attivita': (start_time + timedelta(minutes=duration)).strftime('%Y-%m-%d %H:%M:%S'),
                    'durata_attivita_minuti': duration,
                    'fascia_oraria': 'mattina' if start_time.hour < 11 else 'pranzo' if start_time.hour < 16 else 'cena',
                    'subject_id': subject
                })

    return pd.DataFrame(data)


def generate_sensor_data():
    """
    Genera dati fittizi per i sensori di mobilità.
    Simula rilevazioni PIR in diverse stanze dell'abitazione.
    """
    np.random.seed(44)
    data = []
    start_date = datetime(2024, 1, 1)
    rooms = ['cucina', 'soggiorno', 'camera_letto', 'bagno', 'ingresso']

    for subject in [1, 2, 3]:
        for day in range(180):
            date = start_date + timedelta(days=day)

            for hour in range(6, 23):
                for _ in range(np.random.poisson(2)):
                    timestamp = date + timedelta(hours=hour, minutes=np.random.randint(0, 60))
                    room = np.random.choice(rooms, p=[0.3, 0.25, 0.2, 0.15, 0.1])
                    duration = max(30, int(np.random.exponential(300)))

                    data.append({
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'sensor_id': f'S_{room}_{np.random.randint(1, 4)}',
                        'sensor_type': 'PIR',
                        'room': room,
                        'sensor_status': 'active',
                        'duration_seconds': duration,
                        'subject_id': subject
                    })

    return pd.DataFrame(data)


def generate_all_data(output_dir='data'):
    """
    Genera tutti i dataset e li salva nella directory specificata.

    Args:
        output_dir: Directory dove salvare i CSV (default: 'data')
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generazione dati in corso...")

    sleep_df = generate_sleep_data()
    kitchen_df = generate_kitchen_data()
    sensor_df = generate_sensor_data()

    sleep_df.to_csv('sonno_data.csv', index=False)
    kitchen_df.to_csv('cucina_data.csv', index=False)
    sensor_df.to_csv('sensor_data.csv', index=False)




if __name__ == "__main__":
    generate_all_data()
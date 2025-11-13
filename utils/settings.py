import base64

from dotenv import load_dotenv
import os
from pathlib import Path
from google.api_core import exceptions
import time
from langchain_experimental.utilities.python import PythonREPL
from langchain_ollama import ChatOllama

# Carica le variabili dal file .env
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
python_repl = PythonREPL()
# Recupera la chiave dall'ambiente
google_api_key = os.getenv("GOOGLE_API")
mistral_api = os.getenv("MISTRAL")

ollama_user=os.getenv("OLLAMA_USER")
ollama_pwd=os.getenv("OLLAMA_PWD")
credentials = f"{ollama_user}:{ollama_pwd}"
base64_credentials = base64.b64encode(credentials.encode()).decode()


llm_graph_generator = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=google_api_key,
    temperature=0.7,
    max_retries=0
)


llm_code  = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.3,  # creativo
)

llm_correlation  = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.3,  # creativo
    top_p=0.5,  # varietà lessicale
    top_k=20,  # maggiori opzioni (da vedere meglio)
    max_output_tokens=5096,  #lunghezza risposta (forse anche meno)
    timeout=60.0,
    max_retries=0
)
client_oll = ChatOllama(
    model='qwen3:32b',
    base_url='https://lovelace.ewlab.di.unimi.it/ollama',
    reasoning=False,
    client_kwargs={
            "headers": {
                "Authorization": f"Basic {base64_credentials}"
        }
    }

)
llm_agents = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0,
    top_p=0.1,
    top_k=1,
    max_output_tokens=2048,
    timeout=60.0,
    max_retries=0
)

llm_query = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=google_api_key,
    temperature=0,  #deterministico
    top_p=0.1,  #token più probabili
    top_k=1,  #solo il migliore
    max_output_tokens=1024,  #per json dovrebbe bastare
    timeout=30.0,
    max_retries=0
)

llm_tool = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=google_api_key,
    temperature=0.5,  # Leggermente creativo per intent detection
    max_output_tokens=1024,
)


def invoke_with_retry(agent, messages, max_retries=3):
    retry_count = 0

    while retry_count <= max_retries:
        try:
            result = agent.invoke({"messages": messages})
            return result

        except exceptions.ResourceExhausted as exc:
            retry_count += 1

            if hasattr(exc, 'retry_delay') and exc.retry_delay:
                retry_delay = exc.retry_delay.seconds
            else:

                import re
                match = re.search(r'retry_delay \{\s*seconds: (\d+)', str(exc))
                retry_delay = int(match.group(1)) if match else 60

            if retry_count <= max_retries:
                print(f"Quota exceeded. Waiting {retry_delay} seconds before retry {retry_count}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries ({max_retries}) reached. Raising exception.")
                raise

def invoke_with_structured_output(llm, router,messages, max_retries=3):
    retry_count = 0

    while retry_count <= max_retries:
        try:
            result = llm.with_structured_output(router).invoke(messages)

            return result

        except exceptions.ResourceExhausted as exc:
            retry_count += 1

            if hasattr(exc, 'retry_delay') and exc.retry_delay:
                retry_delay = exc.retry_delay.seconds
            else:

                import re
                match = re.search(r'retry_delay \{\s*seconds: (\d+)', str(exc))
                retry_delay = int(match.group(1)) if match else 60

            if retry_count <= max_retries:
                print(f"Quota exceeded. Waiting {retry_delay} seconds before retry {retry_count}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries ({max_retries}) reached. Raising exception.")
                raise


PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"


SLEEP_DATA_PATH = DATA_DIR / "sonno_data.csv"
KITCHEN_DATA_PATH = DATA_DIR / "cucina_data.csv"
SENSOR_DATA_PATH = DATA_DIR / "sensor_data.csv"


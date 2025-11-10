import base64
from typing import List
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from tool import get_sleep_data, get_kitchen_data

load_dotenv()
ollama_user=os.getenv("OLLAMA_USER")
ollama_pwd=os.getenv("OLLAMA_PWD")
credentials = f"{ollama_user}:{ollama_pwd}"
base64_credentials = base64.b64encode(credentials.encode()).decode()

@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True





if __name__ == "__main__":
    llm = ChatOllama(
        model='qwen3:32b',
        base_url='https://lovelace.ewlab.di.unimi.it/ollama',
        reasoning=False,
        validate_model_on_init=True,
        client_kwargs={
            "headers": {
                "Authorization": f"Basic {base64_credentials}"
            }
        }
    ).bind_tools([get_sleep_data, get_kitchen_data])
    result = llm.invoke(
        "come ha dormtio il soggetto 2 negli ultimi 10 giorni"
    )
    print(result)



    if isinstance(result, AIMessage) and result.tool_calls:
        print(result.tool_calls)
import base64
from typing import List
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from tool import get_sleep_data, get_kitchen_data

from langsmith import Client



if __name__ == "__main__":
    client = Client()

    run_id = ""
    result = client.list_runs(project_id="16a937f5-deab-4612-9902-fc67c7506aaa")

    for run in result:
        if run.status == "pending":
            print(f"Run ID: {run.id}")
            print(f"Name: {run.name}")
            print(f"Status: {run.status}")
            print(f"Start time: {run.start_time}")
            print("-" * 50)
            client.update_run(run_id=run.id, status="aborted")


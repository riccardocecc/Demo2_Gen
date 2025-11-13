from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from langgraph.graph import END
from utils.settings import client_oll as llm_code
from type import State


def create_conversational_node():
    system_message = """
        /no thinking 
        Sei un assistente per un sistema di analisi dati sanitari.


    Il tuo compito è decidere come gestire la richiesta dell'utente:

    1. **Scegli "extract_data"** SE l'utente richiede un'analisi dati su:
       - Sonno
       - Cucina
       - Mobilità
       - Correlazioni tra dati

       IMPORTANTE: Prima di scegliere "extract_data", VERIFICA che siano presenti:
       - ID del soggetto (OBBLIGATORIO)
       - Dominio di analisi: sonno/cucina/mobilità (OBBLIGATORIO)
       - Periodo temporale (OPZIONALE)

       Se mancano informazioni obbligatorie, scegli "END" e chiedi all'utente i dati mancanti.

    2. **Scegli "FINISH"** SE l'utente:
       - Saluta o ringrazia
       - Chiede aiuto o informazioni generali ("cosa sai fare?", "come funziona?")
       - Fa domande che non richiedono analisi dati
       - Ha fornito una richiesta di analisi INCOMPLETA (mancano dati obbligatori)

       Quando scegli FINISH, fornisci una risposta conversazionale, utile e amichevole.

    REGOLE:
    - Rispondi sempre in modo naturale e diretto
    - Non mostrare il tuo ragionamento interno
    - Se non sei sicuro, è meglio scegliere FINISH e chiedere chiarimenti"""


    class RouteSchema(BaseModel):
        response: str = Field(
            description="Risposa conversazionale da invare all'utente, Deve essre sempre completa e utile"
        )
        next: Literal["extract_data", END] = Field(
            description="Scegli 'extract_data' se la richiesta è completa per un'analisi dati, 'END' per rispondere all'utente direttamente "
        )

    def route(state: State) -> Command[Literal['extract_data', '__end__']]:
        messages = [{"role":"system", "content": system_message}]

        for msg in state["messages"]:
            messages.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content
            })

        decision = llm_code.with_structured_output(RouteSchema).invoke(messages,reasoning=False)

        print("Decisione presa",decision.next)

        ai_message = AIMessage(content=decision.response)

        if decision.next == "END":
            return Command(
                goto="__end__",
                update={"messages": [ai_message]}
            )

        return Command(
            goto="extract_data",
        )
    return route
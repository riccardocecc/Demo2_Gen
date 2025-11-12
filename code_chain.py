from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, List, Union
from settings import llm_code


class Code(BaseModel):
    """Schema per il codice generato."""
    description: str = Field(description="Brevissima descrizione del codice")
    imports: str = Field(description="Blocco di import statements")
    code: str = Field(description="Blocco di codice senza imports")


def create_system_instructions(
        context: Optional[str] = None,
        result_var: str = "result",
        result_format: str = "as a dictionary"
) -> str:
    """Crea le istruzioni di sistema per la generazione del codice."""

    instructions = f"""You are a Python coding expert.
You MUST respond using the `code` tool with valid JSON.

Response schema:
{{
  "description": "brief code explanation",
  "imports": "import statements only",
  "code": "implementation code that stores final result in variable '{result_var}' {result_format}"
}}

Rules:
- Provide executable code with all dependencies
- Separate imports from implementation
- No natural language outside the tool response"""

    if context:
        instructions += f"\n\n<context>\n{context}\n</context>"

    return instructions


def create_code_chain(
        context: Optional[str] = None,
        result_var: str = "result",
        result_format: str = "as a dictionary",
        max_retries: int = 3
):
    """
    Crea una chain completa per la generazione di codice con retry automatico.

    Args:
        context: Contesto aggiuntivo per il prompt
        result_var: Nome della variabile risultato (default: 'result')
        result_format: Formato del risultato (default: 'as a dictionary')
        max_retries: Numero massimo di retry (default: 3)

    Returns:
        Chain configurata
    """

    system_instructions = create_system_instructions(context, result_var, result_format)
    structured_llm = llm_code.with_structured_output(Code, include_raw=True)

    def format_messages(inputs: dict) -> List:
        """Formatta i messaggi per l'LLM."""
        messages = [{"role": "system", "content": system_instructions}]

        # Aggiungi i messaggi dell'utente/assistente
        for msg in inputs.get("messages", []):
            if isinstance(msg, (HumanMessage, AIMessage)):
                messages.append({
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content
                })
            elif isinstance(msg, tuple):
                role = "user" if msg[0] == "human" else "assistant"
                messages.append({"role": role, "content": msg[1]})

        return messages

    def validate_output(output: dict) -> dict:
        """Valida l'output dell'LLM e solleva eccezioni se necessario."""

        # Debug log
        if raw := output.get("raw"):
            try:
                print(f"[DEBUG] Raw content: {raw.content[:200]}...")
            except Exception:
                pass

        # Controlla errori di parsing
        if parsing_error := output.get("parsing_error"):
            raise ValueError(
                f"Parsing error: {parsing_error}\n"
                f"LLM must invoke the 'code' tool with valid JSON."
            )

        # Verifica presenza del parsed object
        if not (parsed := output.get("parsed")):
            raise ValueError(
                "Tool not invoked or parsing failed.\n"
                "You MUST use the 'code' tool to structure your response."
            )

        return output

    def handle_retry(inputs: dict) -> dict:
        """Gestisce i retry aggiungendo il messaggio di errore."""
        error = inputs["error"]
        messages = list(inputs["messages"])

        messages.append((
            "assistant",
            f"RETRY REQUIRED:\n{error}\n\n"
            "Fix the error and invoke the 'code' tool correctly."
        ))

        return {"messages": messages}

    def extract_parsed(output: dict) -> Code:
        """Estrae l'oggetto parsed dal risultato."""
        return output["parsed"]

    # Chain principale
    main_chain = (
            format_messages
            | structured_llm
            | validate_output
    )

    # Chain con retry
    retry_chain = handle_retry | main_chain

    # Configura fallback
    chain_with_retries = main_chain.with_fallbacks(
        fallbacks=[retry_chain] * max_retries,
        exception_key="error"
    )

    # Aggiungi estrazione finale
    return chain_with_retries | extract_parsed



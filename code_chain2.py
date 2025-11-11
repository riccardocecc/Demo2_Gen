from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from settings import llm_code

class code(BaseModel):
    description: str = Field(description="Brevissima descrizone del codice")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not include import statements")


def create_code_gen_prompt(
    context: Optional[str] = None,
    result_var: str = "result",
    result_format: str = "as a dictionary"
):
    """
    Crea un prompt per la generazione di codice con contesto opzionale.

    Args:
        context: Contesto aggiuntivo da includere nel prompt
        result_var: Nome della variabile in cui salvare il risultato (default: 'result')
        result_format: Formato del risultato (default: 'as a dictionary')

    Returns:
        ChatPromptTemplate configurato
    """
    base_instructions = f"""/no thinking <instructions>  
You are a coding assistant with expertise in Python.  
You MUST invoke the provided tool named `code` to return your answer.  
DO NOT write natural language explanations outside the tool.  
Your response MUST be a valid JSON matching this schema:

{{{{
  "description": "short explanation of the code purpose",
  "imports": "code block with import statements",
  "code": "code block implementing the solution (without imports). Store the final result in a variable named '{result_var}' {result_format}."
}}}}

Ensure that the code you provide is executable, with all required imports, data definitions, and dependencies included."""

    if context:
        system_message = f"{base_instructions}\n\n<context>\n{context}\n</context>\n\nHere is the user question:"
    else:
        system_message = f"{base_instructions}\n\nHere is the user question:"

    return ChatPromptTemplate.from_messages(
        [
            ("human", system_message),
            ("placeholder", "{messages}"),
        ]
    )
structured_llm = llm_code.with_structured_output(code, include_raw=True)


def check_llm_output(tool_output):
    """Verifica parse error o mancata invocazione del tool e ritorna il parsed object."""
    raw = tool_output.get("raw")
    if raw is not None:
        try:
            print("RAW CONTENT (for debug):\n", raw.content)
        except Exception:
            pass

    if tool_output.get("parsing_error"):
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Parse error: {error}. RAW: {raw}"
        )

    parsed = tool_output.get("parsed")
    if not parsed:
        print("Tool calls:", tool_output.get("tool_calls"))
        raise ValueError(
            "You did not use the provided tool or parsing failed! Be sure to invoke the tool to structure the output."
        )

    return tool_output


def create_code_chain_2(
    context: Optional[str] = None,
    result_var: str = "result",
    result_format: str = "as a dictionary"
):
    """
    Crea una chain completa per la generazione di codice con retry logic.

    Args:
        context: Contesto aggiuntivo da includere nel prompt
        result_var: Nome della variabile in cui salvare il risultato (default: 'result')
        result_format: Formato del risultato (default: 'as a dictionary')

    Returns:
        Chain configurata e pronta all'uso
    """
    print(context)
    code_gen_prompt = create_code_gen_prompt(context, result_var, result_format)

    code_chain_raw = code_gen_prompt | structured_llm | check_llm_output

    def insert_errors(inputs):
        """Insert errors for tool parsing in the messages"""
        error = inputs["error"]
        messages = inputs["messages"]
        messages += [
            (
                "assistant",
                f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool."
            )
        ]
        return {
            "messages": messages,
        }

    fallback_chain = insert_errors | code_chain_raw
    N = 3
    code_chain_re_try = code_chain_raw.with_fallbacks(
        fallbacks=[fallback_chain] * N, exception_key="error"
    )

    def parse_output(solution):
        return solution["parsed"]

    return code_chain_re_try | parse_output
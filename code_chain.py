from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from settings import client_oll, llm_code
from typing_extensions import TypedDict
from langchain_experimental.utilities.python import PythonREPL

python_repl = PythonREPL()
import pandas as pd


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
        context : Additional context for code generation
    """
    error: str
    messages: List
    generation: str
    iterations: int
    context: Optional[str]  # Aggiungi context


class code(BaseModel):
    description: str = Field(description="Brevissima descrizone del codice")
    imports: str = Field(description="Code block import statements")
    code: str = Field(
        description="Code block not include import statements. Store results in a variable named 'result' as a dictionary")


def create_code_gen_prompt(context: Optional[str] = None):
    """
    Crea un prompt per la generazione di codice con contesto opzionale.

    Args:
        context: Contesto aggiuntivo da includere nel prompt (es. schema del DataFrame, variabili disponibili, ecc.)

    Returns:
        ChatPromptTemplate configurato
    """
    base_instructions = """/no thinking <instructions>  
You are a coding assistant with expertise in Python for statistical analysis.  
You MUST invoke the provided tool named `code` to return your answer.  
DO NOT write natural language explanations outside the tool.  
Your response MUST be a valid JSON matching this schema:

You do NOT need to import or create it. Just use it directly in your code.
{{
  "description": "short explanation of the code purpose",
  "imports": "code block with import statements",
  "code": "code block implementing the solution (without imports). Store the final result in a variable named 'result' as a dictionary."
}}

Ensure that the code you provide is executable, with all required imports, data definitions, and dependencies included."""

    # Aggiungi il contesto se fornito
    if context:
        system_message = f"{base_instructions}\n\n<context>\n{context}\n</context>\n\nHere is the user question:"
    else:
        system_message = f"{base_instructions}\n\nHere is the user question:"

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("placeholder", "{messages}"),
        ]
    )


structured_llm = client_oll.with_structured_output(code, include_raw=True)


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


def create_code_chain(context: Optional[str] = None):
    """
    Crea una chain completa per la generazione di codice con retry logic.

    Args:
        context: Contesto aggiuntivo da includere nel prompt

    Returns:
        Chain configurata e pronta all'uso
    """
    code_gen_prompt = create_code_gen_prompt(context)
    print("code_gen_prompt",code_gen_prompt)
    code_chain_raw = (
            code_gen_prompt | structured_llm | check_llm_output
    )

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


max_iterations = 3


def generate(state: GraphState):
    """
    Generate a code solution.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """
    print("--Generating code solution--")
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]
    context = state.get("context")  # Recupera il contesto dallo state

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]

    # Crea la chain con il contesto
    code_gen_chain = create_code_chain(context)

    code_solution = code_gen_chain.invoke(
        {"messages": messages}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.description} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


df = pd.DataFrame({
    'A': [1, 2, 3,4,5,6,7,8,9,10],
    'B': [4, 5, 3,4,5,6,7,9,10,11]
})


def code_check(state: GraphState):
    print("--Checking code solution--")
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    imports = code_solution.imports
    code = code_solution.code

    try:
        print("---CHECK IMPORTS---")
        python_repl.run(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the import test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    try:
        print("---CHECK CODE BLOCK---")
        python_repl.globals['df'] = df
        python_repl.run(code)
        result = python_repl.locals.get('result')
        print(f"Result: {result}")
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    print("---CODE CHECK: SUCCESS---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


def decide_to_finish(state: GraphState):
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        return "generate"


def graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("generate", generate)
    workflow.add_node("check_code", code_check)

    # Build graph
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "check_code")
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "generate": "generate",
        },
    )
    return workflow.compile()


if __name__ == "__main__":
    app = graph()

    # Esempio 1: Senza contesto
    question = "Calcola la media del dataframe"
    solution = app.invoke({
        "messages": [("user", question)],
        "iterations": 0,
        "error": "",
        "context": None
    })

    print(solution['messages'])

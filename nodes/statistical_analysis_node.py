
from type import State, StatisticalMethodSelection
from registry.DomainRegistry import domain_registry
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from utils.code_chain import create_code_chain
from utils.settings import client_oll as llm_code

def statistical_analysis_node(state: State) -> State:
    """
    Nodo 2: Genera codice Python per analisi statistica usando Pydantic structured output.
    """
    print("--Generating code solution--")

    messages = state.get("messages", [])
    iterations = state.get("iterations", 0)
    error = state.get("error", "")
    domains_detected = state.get("domains_detected", [])
    available_columns_by_df = domain_registry.get_available_columns_for_domains(domains_detected)

    query = state["query"]
    method_selection = state.get("statistical_method", {})
    raw_data = state.get("raw_data", {})

    available_dataframes = list(raw_data.keys())
    subject = state["subject_id"]

    if error == "yes":
        messages = messages + [
            HumanMessage(
                content="Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:")
        ]

    # Formatta il context come testo leggibile invece di dizionari Python grezzi
    context = f"""Query: "{query}"

Analysis Details:
- Analysis Goal: {method_selection.get('analysis_goal', 'N/A')}
- Analysis Type: {method_selection.get('analysis_type', 'N/A')}
- Variables: {', '.join(method_selection.get('variables', []))}
- Statistical Methods: {', '.join(method_selection.get('statistical_methods', []))}
- Calculations Needed: {', '.join(f"{k}: {v}" for k, v in method_selection.get('calculations_needed', {}).items())}
- Expected Outputs: {', '.join(method_selection.get('expected_outputs', []))}

Available columns:
{chr(10).join(f"- {df_name}: {', '.join(f'{col} ({dtype})' for col, dtype in columns.items())}" for df_name, columns in available_columns_by_df.items())}

Don't filter data by subject. Data already filtred for subject: {subject}

IMPORTANT:
A pandas DataFrame named is already available in the environment:
0. 'data' and 'timestamp_picco' columns are STRINGS - you MUST convert them to dates using pd.to_datetime() BEFORE merging !!!
1. Use ONLY these DataFrames: {', '.join(available_dataframes)}  
2. Access dataframes using their EXACT names: {', '.join(available_dataframes)} 
3. Use ONLY these variables in your analysis: {', '.join(method_selection.get('variables', []))}
4. DO NOT filter data by dates!! - the data is already filtered for the correct time period
5. Work with ALL rows in the DataFrames provided
6. NO GRAPH 
"""

    code_gen_chain = create_code_chain(
        context=context,
        result_var="in globals variable result",
        result_format="as a dictionary"
    )

    code_solution = code_gen_chain.invoke({"messages": messages})

    assistant_message = AIMessage(
        content=f"{code_solution.description}\n\nImports:\n{code_solution.imports}\n\nCode:\n{code_solution.code}"
    )

    iterations = iterations + 1

    return {
        **state,
        "generation": code_solution,
        "messages": messages + [assistant_message],
        "iterations": iterations
    }

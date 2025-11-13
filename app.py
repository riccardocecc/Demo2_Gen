import streamlit as st
from datetime import datetime
import plotly.graph_objects as go

from prova_tool import create_sleep_analysis_graph

# Configurazione pagina
st.set_page_config(
    page_title="Demo Multi Agent - Only Gen",
    page_icon="",
    layout="wide"
)

# Inizializza session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = create_sleep_analysis_graph()


st.title("Demo Multi Agent - Only Gen")
st.markdown("Analisi statistiche piu approfondite")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


        if "plotly_figure" in message:
            try:
                fig = go.Figure(message["plotly_figure"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Errore nella visualizzazione del grafico: {e}")

# Input utente
if prompt := st.chat_input("Es: C'e correlazione tra risvegli e frequenza cardiaca negli ultimi 10 giorni?"):

    # Mostra messaggio utente
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Mostra spinner mentre elabora
    with st.chat_message("assistant"):
        with st.spinner("Analizzo i dati..."):

            # Esegui analisi
            initial_state = {
                "query": prompt,
                "subject_id": 0,
                "period": "",
                "raw_data": {},
                "domains_detected": [],
                "statistical_method": {},
                "messages": [],
                "error": "",
                "iterations": 0,
                "generation": "",
                "code_response": "",
                "plotly_figure": {},
                "plotly_figure_dict": {},
                "plot_attempts": 0,
                "plot_errors": []
            }

            final_state = st.session_state.graph.invoke(initial_state)

            # Gestisci errore
            if final_state.get("error") and final_state.get("error") != "no":
                response = f"Si e verificato un errore durante l'analisi"
                st.error(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            else:
                # Costruisci la risposta basata sui risultati
                code_response = final_state.get("code_response", {})
                statistical_method = final_state.get("statistical_method", {})

                response_parts = []

                # Aggiungi l'obiettivo dell'analisi
                if statistical_method.get("analysis_goal"):
                    response_parts.append(f"**Analisi:** {statistical_method['analysis_goal']}")

                # Aggiungi i risultati statistici
                response = "\n".join(response_parts) if response_parts else "Analisi completata"
                st.markdown(response)

                # Mostra grafico se presente
                plotly_figure_dict = final_state.get("plotly_figure_dict")
                if plotly_figure_dict:
                    try:
                        fig = go.Figure(plotly_figure_dict)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Errore nella visualizzazione del grafico: {e}")

                # Salva in session state
                message_data = {
                    "role": "assistant",
                    "content": response
                }

                if plotly_figure_dict:
                    message_data["plotly_figure"] = plotly_figure_dict

                st.session_state.messages.append(message_data)

# Sidebar con info
with st.sidebar:
    st.header("Informazioni")
    st.markdown("""
    **Esempi di domande:**
    - Mostrami il trend dei risvegli notturni negli ultimi 20 giorni per il soggetto 2
    - C'è correlazione tra il numero di risvegli e l'utilizzo della cucina negli ultimi 20 giorni per il soggetto 2?
    - In quali momenti della giornata il soggetto 2 cucina di più?
    - Confronta il numero di risvegli con il tempo totale di sonno
    """)

    st.markdown("---")
    st.caption(f"Messaggi nella chat: {len(st.session_state.messages)}")

    # Pulsante per pulire la chat
    if st.button("Pulisci chat"):
        st.session_state.messages = []
        st.rerun()
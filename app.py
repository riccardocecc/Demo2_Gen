import streamlit as st
from datetime import datetime

from prova_tool import create_sleep_analysis_chain

# Configurazione pagina
st.set_page_config(
    page_title="Demo Multi Agent - Only Gen",
    page_icon="",
    layout="wide"
)

# Inizializza session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = create_sleep_analysis_chain()

# Header
st.title("Demo Multi Agent - Only Gen")
st.markdown("Analisi statistiche più approfondite")

# Mostra chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Se c'è un grafico, mostralo
        if "plot_html" in message:
            st.components.v1.html(message["plot_html"], height=500, scrolling=True)

# Input utente
if prompt := st.chat_input("Es: C'è correlazione tra risvegli e frequenza cardiaca negli ultimi 10 giorni?"):

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
                "statistical_method": {},
                "analysis_code": "",
                "analysis_results": {},
                "plot_code": "",
                "plot_html": "",
                "final_response": "",
                "messages": [],
                "error": ""
            }

            final_state = st.session_state.chain.invoke(initial_state)

            # Gestisci errore
            if final_state.get("error"):
                response = f"❌ Si è verificato un errore:\n\n{final_state['error']}"
                st.error(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            else:
                # Mostra risposta
                response = final_state.get("final_response", "Analisi completata")
                st.markdown(response)

                # Mostra grafico se presente
                if final_state.get("plot_html"):
                    st.components.v1.html(
                        final_state["plot_html"],
                        height=500,
                        scrolling=True
                    )

                # Salva in session state
                message_data = {
                    "role": "assistant",
                    "content": response
                }

                if final_state.get("plot_html"):
                    message_data["plot_html"] = final_state["plot_html"]

                st.session_state.messages.append(message_data)

# Sidebar con info
with st.sidebar:
    st.header("Informazioni")
    st.markdown("""
    **Esempi di domande:**
    - C'è correlazione tra risvegli e frequenza cardiaca?
    - Qual è la media del sonno REM negli ultimi 7 giorni?
    - Mostra l'andamento del sonno profondo
    - Confronta il numero di risvegli con il tempo totale di sonno
    """)

    st.markdown("---")
    st.caption(f"Messaggi nella chat: {len(st.session_state.messages)}")
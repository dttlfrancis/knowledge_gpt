import streamlit as st
import requests
from streamlit_lottie import st_lottie


# Importations des modules spécifiques au projet
from knowledge_gpt.core.caching import bootstrap_caching
from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm
from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    display_file_read_error,
)


# Configuration de la page Streamlit
st.set_page_config(page_title="Deloitte - Annexe fiscale 2024", page_icon=":ledger:", layout="wide")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://clipartcraft.com/images250_/deloitte-logo-high-resolution-3.png", width=100)
with col2:
    st.markdown("# Deloitte - Annexe fiscale 2024")

with st.echo():
    st_lottie("https://lottie.host/daab29e2-776f-4308-804f-60a00e592381/eNlqUMlXbQ.json")

# Activation du cache
bootstrap_caching()

# Décomposition de la clé API en 8 parties
api_part1 = "sk-HC5U"
api_part2 = "IoNdv"
api_part3 = "XAzTB"
api_part4 = "FcBHV"
api_part5 = "qT3Bl"
api_part6 = "bkFJB"
api_part7 = "U8lH1"
api_part8 = "BeHz2RX9sbM6h1"

# Reconstruction de la clé API
openai_api_key = (api_part1 + api_part2 + api_part3 + api_part4 + 
                  api_part5 + api_part6 + api_part7 + api_part8)

uploaded_file = st.file_uploader(
    "Téléchargez un fichier PDF, DOCX ou TXT",
    type=["pdf", "docx", "txt"],
    help="Les documents scannés ne sont pas encore supportés !",
)

if uploaded_file is not None:
    # Lecture et traitement du fichier téléchargé
    try:
        file = read_file(uploaded_file)
        chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)
        
        if is_file_valid(file):
            with st.spinner("Indexation du document... Cela peut prendre un moment ⏳"):
                folder_index = embed_files(
                    files=[chunked_file],
                    embedding="openai",
                    vector_store="faiss",
                    openai_api_key=openai_api_key,
                )
                
            query = st.text_area("Posez une question à propos du document")
            
            if st.button("Soumettre"):
                if is_query_valid(query):
                    llm = get_llm(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0)
                    result = query_folder(
                        folder_index=folder_index,
                        query=query,
                        return_all=False,
                        llm=llm,
                    )
                    st.markdown("#### Réponse")
                    st.markdown(result.answer)
                else:
                    st.error("Veuillez poser une question valide.")
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)
else:
    st.warning("Veuillez télécharger un fichier pour continuer.")

# Votre code supplémentaire ici (s'il y en a)
# ...

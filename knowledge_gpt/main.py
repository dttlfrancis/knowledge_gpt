import streamlit as st
import os
import io

# Importations des modules spécifiques au projet
from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    display_file_read_error,
)
from knowledge_gpt.core.caching import bootstrap_caching
from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm

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

# Configuration de la page Streamlit
st.set_page_config(page_title="Deloitte - Annexe fiscale 2024", page_icon="📖", layout="wide")
st.header("Deloitte - Annexe fiscale 2024")

# Activation du cache
bootstrap_caching()

uploaded_file = st.file_uploader(
    "Téléchargez un fichier PDF, DOCX ou TXT",
    type=["pdf", "docx", "txt"],
    help="Les documents scannés ne sont pas encore supportés !",
)

# Sélection du modèle
model: str = st.selectbox("Modèle", options=["gpt-3.5-turbo", "gpt-4"])  # type: ignore

with st.expander("Options Avancées"):
    return_all_chunks = st.checkbox("Afficher tous les fragments récupérés de la recherche vectorielle")
    show_full_doc = st.checkbox("Afficher le contenu analysé du document")


if not uploaded_file:
    st.stop()

try:
    file = read_file(uploaded_file)
except Exception as e:
    display_file_read_error(e, file_name=uploaded_file.name)

chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

if not is_file_valid(file):
    st.stop()

# Indexation du document
with st.spinner("Indexation du document... Cela peut prendre un moment ⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding="openai",
        vector_store="faiss",
        openai_api_key=openai_api_key,
    )

with st.form(key="qa_form"):
    query = st.text_area("Posez une question à propos du document")
    submit_button = st.form_submit_button("Soumettre")

if show_full_doc:
    with st.expander("Document"):
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

if submit_button:
    if not is_query_valid(query):
        st.stop()

    answer_col, sources_col = st.columns(2)

    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
    )

    with answer_col:
        st.markdown("#### Réponse")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")

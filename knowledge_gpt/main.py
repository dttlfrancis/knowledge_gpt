import streamlit as st
import os

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

# Paramètres de configuration
EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

# Clé API OpenAI intégrée
openai_api_key = "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"

# Configuration de la page Streamlit
st.set_page_config(page_title="Deloitte - Annexe fiscale 2024", page_icon="📖", layout="wide")
st.header("Deloitte - Annexe fiscale 2024")

# Amélioration de l'esthétique de la page
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Activation du cache
bootstrap_caching()

uploaded_file = st.file_uploader(
    "Téléchargez un fichier PDF, DOCX ou TXT",
    type=["pdf", "docx", "txt"],
    help="Les documents scannés ne sont pas encore supportés !",
)

model: str = st.selectbox("Modèle", options=MODEL_LIST, index=1)  # type: ignore

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
        embedding=EMBEDDING,
        vector_store=VECTOR_STORE,
        openai_api_key=openai_api_key,
    )

with st.form(key="qa_form"):
    query = st.text_area("Posez une question à propos du document")
    submit = st.form_submit_button("Soumettre")

if show_full_doc:
    with st.expander("Document"):
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

if submit:
    if not is_query_valid(query):
        st.stop()

    colonne_reponse, colonne_sources = st.columns(2)

    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
    )

    with colonne_reponse:
        st.markdown("#### Réponse")
        st.markdown(result.answer)

    with colonne_sources:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")

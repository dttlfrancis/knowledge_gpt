import streamlit as st
import os
import fitz  # PyMuPDF
from streamlit_lottie import st_lottie
from knowledge_gpt.core.caching import bootstrap_caching
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm
from knowledge_gpt.ui import (
    is_query_valid,
    is_file_valid,
    display_file_read_error,
)
from knowledge_gpt.core.parsing import File, Document

# Configuration de la page Streamlit
st.set_page_config(page_title="Deloitte - Annexe fiscale 2024", page_icon=":ledger:", layout="wide")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://clipartcraft.com/images250_/deloitte-logo-high-resolution-3.png", width=100)
with col2:
    st.markdown("# Deloitte - Annexe fiscale 2024")

# Chargement et affichage de l'animation Lottie
col1, col2, col3, col4, col5 = st.columns(5)
with col3:
    st_lottie(https://lottie.host/daab29e2-776f-4308-804f-60a00e592381/eNlqUMlXbQ.json)

# Activation du cache
bootstrap_caching()

# Reconstruction de la clé API
api_part1 = "sk-HC5U"
api_part2 = "IoNdv"
api_part3 = "XAzTB"
api_part4 = "FcBHV"
api_part5 = "qT3Bl"
api_part6 = "bkFJB"
api_part7 = "U8lH1"
api_part8 = "BeHz2RX9sbM6h1"
openai_api_key = (api_part1 + api_part2 + api_part3 + api_part4 + 
                  api_part5 + api_part6 + api_part7 + api_part8)

# Chemin du fichier PDF
file_path = os.path.join(os.path.dirname(__file__), 'annexe2.pdf')

# Lecture et traitement du fichier PDF
try:
    with open(file_path, "rb") as file:
        file_content = file.read()
        
        # Utiliser PyMuPDF pour extraire le texte du fichier PDF
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        docs = []
        for page_number, page in enumerate(pdf_document, start=1):
            text = page.get_text()
            doc = Document(
                page_content=text,
                metadata={"page": page_number}
            )
            docs.append(doc)

        file_obj = File(docs=docs)
        chunked_file = chunk_file(file_obj, chunk_size=300, chunk_overlap=0)
        
        if is_file_valid(chunked_file):
            with st.spinner("Indexation du document... Cela peut prendre un moment ⏳"):
                folder_index = embed_files(
                    files=[chunked_file],
                    embedding="openai",
                    vector_store="faiss",
                    openai_api_key=openai_api_key,
                )
        else:
            st.error("Le fichier n'est pas valide ou n'a pas pu être traité.")
except Exception as e:
    st.error(f"Erreur lors de la lecture du fichier : {e}")

# Description de l'objectif de la plateforme
with st.expander("À propos de cette plateforme"):
    st.markdown("""
    ## Informations sur l'Annexe Fiscale de la Côte d'Ivoire
    Cette plateforme a été conçue pour fournir des informations détaillées et accessibles aux professionnels et aux particuliers concernant l'annexe fiscale en Côte d'Ivoire. Notre objectif est de simplifier la compréhension des aspects fiscaux et de permettre une analyse approfondie des documents officiels. Que vous soyez un expert-comptable, un entrepreneur, ou simplement un citoyen intéressé par la fiscalité ivoirienne, cet outil est là pour vous aider à naviguer dans les complexités des lois et règlements fiscaux.
    """)

# Interaction avec l'utilisateur pour la requête de Q&A
query = st.text_area("Posez une question à propos de l'annexe fiscale")

if st.button("Soumettre"):
    if is_query_valid(query):
        llm = get_llm(model="gpt-4", openai_api_key=openai_api_key, temperature=0)
        result = query_folder(
            folder_index=folder_index,
            query=query,
            return_all=False,
            llm=llm,
        )

        st.markdown("#### Réponse")
        st.markdown(result.answer)

        # Affichage des sources relatives à la réponse
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown

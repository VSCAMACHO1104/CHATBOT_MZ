#importación de modulos de las librerías

import requests
import streamlit as st
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

#Configuración visual de la página web con streamlit

st.set_page_config( menu_items={"about":"This page was created by Victor Camacho G. for contact mail to vsmacho@gmail.com"},page_title="CHATBOT_MZ",page_icon="https://raw.githubusercontent.com/VSCAMACHO1104/CHATBOT_MZ/main/logonavegadorCHATBOTMZ.png?token=GHSAT0AAAAAACVHWV62QDHHXHB6GZ4ASY3AZVAZZDQ")
image_id='1zfFcvWC7L05vXZCcgZ4mgK7qmTna6ewP'
image_url=f'https://drive.google.com/uc?export=view&id={image_id}'
response= requests.get(image_url)
st.image(response.content,caption='Powered by OpenAI ֎', use_column_width=True)
st.header("CONSULTA DE PROCEDIMIENTOS ADUANEROS")
st.markdown(
    """
    <style>
    .center-text {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="center-text">🛃📑🔍</div>', unsafe_allow_html=True)

#Input de recursos (doc pdf y contraseña) en streamlit
OPENAI_API_KEY = st.text_input('OpenAI API Key', type='password')
pdf_obj = st.file_uploader("Carga tu documento", type="pdf", on_change=st.cache_resource.clear)

col1,col2=st.columns([2,4],gap="small")
col1.image("https://www.tramitesmz.com/images/logos/Logo_Principal.jpg")
col2.image("https://www.hacienda.go.cr/docs/LogoMinisteriodeHaciendacolor.png")

#Configuración de los embeddings y BD vectorial

@st.cache_resource 
def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )        
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

if pdf_obj:
    knowledge_base = create_embeddings(pdf_obj)
    user_question = st.text_input("Realiza una consulta sobre el documento:")

    if user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        docs = knowledge_base.similarity_search(user_question, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)

        st.write(respuesta)    
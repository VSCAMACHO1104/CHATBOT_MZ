#importación de modulos de las librerías

import requests
import streamlit as st
from streamlit_lottie import st_lottie
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback

#Configuración visual de la página web con streamlit (Encabezados, subtitulos e imagenes)

st.set_page_config( menu_items={"about":"This page was created by Victor Camacho G. for contact mail to vsmacho@gmail.com"}, page_title="CHATBOT_MZ",page_icon="https://raw.githubusercontent.com/VSCAMACHO1104/CHATBOT_MZ/main/logonavegadorCHATBOTMZ.png?token=GHSAT0AAAAAACVHWV62QDHHXHB6GZ4ASY3AZVAZZDQ")
image_id='1zfFcvWC7L05vXZCcgZ4mgK7qmTna6ewP'
image_url=f'https://drive.google.com/uc?export=view&id={image_id}'
response= requests.get(image_url)
st.image(response.content,caption='Powered by OpenAI ֎', use_column_width=True)
st.header("CONSULTA DE PROCEDIMIENTOS ADUANEROS")
st.title("🛃📑🔍")

# Inicializar el estado de sesión para acumular tokens y costos
if 'total_tokens_accum' not in st.session_state:
    st.session_state.total_tokens_accum = 0
if 'total_cost_accum' not in st.session_state:
    st.session_state.total_cost_accum = 0

#Input de recursos (doc pdf y contraseña API Key) y siguientes imagenes en streamlit 

OPENAI_API_KEY = st.text_input('OpenAI API Key', type='password')
col1,col2,col3=st.columns([6,5,5],gap="small")
with col2:
    st_lottie("https://lottie.host/1c8210b1-d77c-4c4f-b027-bb36de2b26aa/w81pWXFvwu.json",width=150, height=150)
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


        with get_openai_callback() as cb:
            respuesta = chain.run(input_documents=docs, question=user_question)
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            total_tokens = cb.total_tokens
            total_cost = cb.total_cost

             # Actualizar los acumulados
            st.session_state.total_tokens_accum += total_tokens
            st.session_state.total_cost_accum += total_cost

            # Mostrar la respuesta en el cuerpo principal
            st.write(respuesta)

            with st.sidebar: 
                st_lottie("https://lottie.host/758306d3-ae78-4808-be51-74c8e175b131/qVMbfJ6Mba.json",width=250, height=250)

                # Mostrar la información de los tokens y el costo en el sidebar
                st.sidebar.title("Uso de la App")             
                st.sidebar.write(f"Tokens del prompt: {prompt_tokens}")
                st.sidebar.write(f"Tokens completados: {completion_tokens}")
                st.sidebar.write(f"Tokens totales: {total_tokens}")
                st.sidebar.write(f"Costo total (USD): ${total_cost:.4f}")  

                # Mostrar el acumulado de tokens y costo en el sidebar
                st.write("## Acumulados")
                st.sidebar.write(f"Tokens totales acumulados: {st.session_state.total_tokens_accum}")
                st.sidebar.write(f"Costo total acumulado (USD): ${st.session_state.total_cost_accum:.4f}")      
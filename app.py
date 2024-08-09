import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from Functions.text_retrieval import generate_response
from Functions.app_functions import load_chat_history, save_chat_history
from Functions.database import faiss_query, faiss_get_vector_store
from Functions.text_retrieval import generate_response, generate_response_historico
from Functions.cuda_test import gpu_test
import os
from datetime import datetime

# Testing GPU conection -------------------------------------------------------------------------------------------------------------------
# gpu_test()

# OpenAI and HuggingFaceHub api key -------------------------------------------------------------------------------------------------------
load_dotenv(find_dotenv())

# Avatars ---------------------------------------------------------------------------------------------------------------------------------
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Variables -------------------------------------------------------------------------------------------------------------------------------
n_results = 8      # Original RAG paper uses n_results = {5, 10}
similarity_function = "ip"      # Similitary functions: "l2" (Squared L2), "ip" (Inner product), "cosine" (Cossine) | Original RAG paper uses "ip"
model_name = "paraphrase-multilingual-MiniLM-L12-v2" # paraphrase-multilingual-MiniLM-L12-v2, paraphrase-multilingual-mpnet-base-v2, 
                                                     # distiluse-base-multilingual-cased-v1, distiluse-base-multilingual-cased-v2
model_str = "Usando o modelo: \n" + model_name

# Models ----------------------------------------------------------------------------------------------------------------------------------
model = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1")
# model = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
# model = HuggingFaceEndpoint(repo_id="google/gemma-7b") 
# model = ChatOpenAI(model_name="gpt-3.5-turbo")
embedding = HuggingFaceEmbeddings(model_name = model_name) # paraphrase-multilingual-MiniLM-L12-v2, paraphrase-multilingual-mpnet-base-v2, 
                                                                                        # distiluse-base-multilingual-cased-v1, distiluse-base-multilingual-cased-v2
# Creating/Loading vector_store -----------------------------------------------------------------------------------------------------------
vector_store = faiss_get_vector_store(embedding, model_name)

# Main interface and response generator ---------------------------------------------------------------------------------------------------
def main():
    st.title("RAG interface test")
    dialogos = ["Novo Diálogo"] + os.listdir("chat_history")

    # Sidebar with a button to delete chat history
    with st.sidebar:
        st.title("Diálogos")
        select = st.selectbox("Selecione um diálogo", dialogos)
        if select == "Novo Diálogo":
            time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            save_chat_history([], time)

        st.text(model_str)
        if st.button("Deletar histórico de mensagens"):
            st.session_state.messages = []
            time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            save_chat_history([], time)

    # Initialize or load chat history
    if "messages" not in st.session_state:
        time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        st.session_state.messages = load_chat_history(time)

    # Display chat messages
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escreva aqui sua mensagem"):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        documents_faiss = faiss_query(vector_store, prompt, n_results, similarity_function)
        documents_str = ""
        for element in documents_faiss:
            documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\testing_docs\\", "")).replace(".md", "") + "\n"
            documents_str += "Conteúdo: " + element.page_content + "\n\n"

        if (len(st.session_state.messages)) < 2:
            historico = "Não há" + "\n"
        else:
            historico = ""
            for message in st.session_state.messages[:-1]:
                if message["role"] == "user":
                    historico += "Pergunta: " + message["content"] + "\n\n"
                if message["role"] == "assistant":
                    historico += message["content"].replace("\n", "") + "\n\n"

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            # response = generate_response(model, documents_str, prompt)
            response = generate_response_historico(model, documents_str, historico, prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Save chat history after each interaction
    time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    save_chat_history(st.session_state.messages, time)

if __name__ == "__main__":
    main()

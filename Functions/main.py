from text_preparation import get_everything, get_chunks_id_csv
from evaluation import eval_embeddings, eval_chunk_size, evaluateAll, evaluateModels, getNumWords, getCorrectChunks
from database import use_chroma, faiss_query, faiss_embed_documents, faiss_get_vector_store
from text_retrieval import generate_response, generate_response_historico
from cuda_test import gpu_test
import numpy as np
from dotenv import load_dotenv, find_dotenv
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
import pandas as pd

# Testing GPU conection -------------------------------------------------------------------------------------------------------------------
# gpu_test()

# Starting timer --------------------------------------------------------------------------------------------------------------------------
# tempo_inicial = time.time()

# Load api keys ---------------------------------------------------------------------------------------------------------------------------
load_dotenv(find_dotenv())

# Variables -------------------------------------------------------------------------------------------------------------------------------
chunk_size = 2000
chunk_overlap = 200
n_results = 3     # Recommended values between {5, 10}
similarity_function = "ip"     # Similitary functions: "l2" (Squared L2), "ip" (Inner product), "cosine" (Cossine) | Original RAG paper uses "ip"
query_text = "Quando a Universidade de São Paulo foi fundada?"
model_name = "paraphrase-multilingual-mpnet-base-v2" # paraphrase-multilingual-MiniLM-L12-v2, paraphrase-multilingual-mpnet-base-v2, 
                                                     # distiluse-base-multilingual-cased-v1, distiluse-base-multilingual-cased-v2

# Text preparation ------------------------------------------------------------------------------------------------------------------------
# text_chunks, ids = get_everything(chunk_size, chunk_overlap)
# get_chunks_id_csv(text_chunks, ids)
# query_text_list = [query_text]

#      Debug Prints
# print(text_chunks[10])
# print(text_chunks[10].metadata)
# print(text_chunks[10].page_content)

# Execution time for text preparation -----------------------------------------------------------------------------------------------------
# tempo_de_execucao_text_preparation = time.time() - tempo_inicial
# print("Tempo de execução para preparar o texto:", tempo_de_execucao_text_preparation, "segundos.")

# Models ----------------------------------------------------------------------------------------------------------------------------------
# model = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1")
# # model = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
# # model = HuggingFaceEndpoint(repo_id="google/gemma-7b")
# # model = ChatOpenAI(model_name="gpt-3.5-turbo")
embedding = HuggingFaceEmbeddings(model_name = model_name)
# retriever = BM25Retriever.from_documents(text_chunks, k = n_results)

# ChromaDB (Not Used) ---------------------------------------------------------------------------------------------------------------------
# context_chroma = use_chroma(text_chunks=text_chunks, similarity_function=similarity_function, n_results=n_results, ids=ids, query_text_list=query_text_list)
# results_array = np.array(context_chroma)
# documents_str = map(str, results_array[0])
# documents_str = ",".join(str(element) for element in documents_str)

# Creating FAISS and doing similarity search ----------------------------------------------------------------------------------------------
# vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
vector_store = faiss_get_vector_store(embedding, model_name)
# result = faiss_query(vector_store, query_text, n_results, similarity_function)

#      Debug Prints
# print(result)
# print(result[0].metadata["id"])

# Okapi BM25 without ElasticSearch --------------------------------------------------------------------------------------------------------
# result = retriever.invoke(query_text)

#      Debug Prints
# print(result)
# print(result[0].metadata["id"])

# Evaluation ------------------------------------------------------------------------------------------------------------------------------
# eval_embeddings(n_results, similarity_function)
# eval_chunk_size(model_name, n_results, similarity_function)
# evaluateAll(n_results, similarity_function)
ks = [3, 5, 8]
for k in ks:
    evaluateModels(vector_store, chunk_size, chunk_overlap, k, similarity_function)

#       Getting number of words for different chunk_sizes ---------------------------

# getNumWords()

#       Getting csv for correct chunks after query ----------------------------------

# ks = [3, 5, 8]
# for k in ks:
#     getCorrectChunks(vector_store, chunk_size, chunk_overlap, k, similarity_function)

# Generate response text ------------------------------------------------------------------------------------------------------------------
# documents_str = ""
# for element in result:
#     documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\md_docs\\", "")).replace(".md", "") + "\n"
#     documents_str += "Conteúdo: " + element.page_content + "\n\n"
# print(documents_str)

# Generate response -----------------------------------------------------------------------------------------------------------------------
# response = generate_response(model=model, documents_str=documents_str, query_text=query_text)
# response = generate_response_historico(model=model, documents_str=documents_str, historico="Não há", query_text=query_text)
# print(response)

# Total execution time --------------------------------------------------------------------------------------------------------------------
# tempo_de_execucao_database = time.time() - tempo_de_execucao_text_preparation - tempo_inicial
# print("Tempo de execução para criar o vector database e gerar resposta:", tempo_de_execucao_database, "segundos.")
# tempo_de_execucao_final = time.time() - tempo_inicial
# print("Tempo de execução total:", tempo_de_execucao_final, "segundos.")

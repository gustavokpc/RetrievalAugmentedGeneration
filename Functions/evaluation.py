import pandas as pd
from database import faiss_query, faiss_embed_documents, faiss_get_vector_store
import random
import matplotlib.pyplot as plt
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from text_preparation import get_everything, get_chunks_id_csv_evaluation
import csv
import numpy as np
from langchain_community.llms import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from text_retrieval import generate_response_evaluation, generate_response_claude, generate_response_maritalk, generate_response_gemini
import anthropic
import maritalk
import os
import google.generativeai as genai
import time

def evaluateAll(n_results, similarity_function):
    chunk_size_list = [2000, 4000, 8000]
    chunk_overlap_list = [200, 400, 800]
    k = list(range(1, 101))
    
    for i in range(len(chunk_size_list)):
        chunk_size = chunk_size_list[i]
        chunk_overlap = chunk_overlap_list[i]
        text_chunks, ids = get_everything(chunk_size, chunk_overlap)
        get_chunks_id_csv_evaluation(text_chunks, ids, chunk_size)

    # MiniLM ----------------------------------------------------------------------------------------------------------
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        embedding = HuggingFaceEmbeddings(model_name = model_name)
        vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
        acc_paraphrase_multilingual_MiniLM_L12_v2 = evaluationVectorStores_chunk_size(vector_store, n_results, similarity_function, chunk_size)

        print("\nModel: " + model_name)
        print("\nAccuracy for chunk_size of", chunk_size, "is ", acc_paraphrase_multilingual_MiniLM_L12_v2)

        filename = "Data/csv_docs/accs_docs/accs_" + model_name + "_" + str(chunk_size) + ".csv"
        with open(filename, 'w', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(zip(k, acc_paraphrase_multilingual_MiniLM_L12_v2))

    # mpnet -----------------------------------------------------------------------------------------------------------
        model_name = "paraphrase-multilingual-mpnet-base-v2"
        embedding = HuggingFaceEmbeddings(model_name = model_name)
        vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
        acc_paraphrase_multilingual_mpnet_base_v2 = evaluationVectorStores_chunk_size(vector_store, n_results, similarity_function, chunk_size)

        print("\nModel: " + model_name)
        print("\nAccuracy for chunk_size of", chunk_size, "is ", acc_paraphrase_multilingual_mpnet_base_v2)

        filename = "Data/csv_docs/accs_docs/accs_" + model_name + "_" + str(chunk_size) + ".csv"
        with open(filename, 'w', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(zip(k, acc_paraphrase_multilingual_mpnet_base_v2))

    # cased_v1 --------------------------------------------------------------------------------------------------------
        model_name = "distiluse-base-multilingual-cased-v1"
        embedding = HuggingFaceEmbeddings(model_name = model_name)
        vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
        acc_distiluse_base_multilingual_cased_v1 = evaluationVectorStores_chunk_size(vector_store, n_results, similarity_function, chunk_size)

        print("\nModel: " + model_name)
        print("\nAccuracy for chunk_size of", chunk_size, "is ", acc_distiluse_base_multilingual_cased_v1)

        filename = "Data/csv_docs/accs_docs/accs_" + model_name + "_" + str(chunk_size) + ".csv"
        with open(filename, 'w', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(zip(k, acc_distiluse_base_multilingual_cased_v1))

    # cased_v2 --------------------------------------------------------------------------------------------------------
        model_name = "distiluse-base-multilingual-cased-v2"
        embedding = HuggingFaceEmbeddings(model_name = model_name)
        vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
        acc_distiluse_base_multilingual_cased_v2 = evaluationVectorStores_chunk_size(vector_store, n_results, similarity_function, chunk_size)

        print("\nModel: " + model_name)
        print("\nAccuracy for chunk_size of", chunk_size, "is ", acc_distiluse_base_multilingual_cased_v2)

        filename = "Data/csv_docs/accs_docs/accs_" + model_name + "_" + str(chunk_size) + ".csv"
        with open(filename, 'w', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(zip(k, acc_distiluse_base_multilingual_cased_v2))

    # BM25 ------------------------------------------------------------------------------------------------------------
        retriever = BM25Retriever.from_documents(text_chunks, k = n_results)
        acc_BM25 = evaluationBM25(retriever, n_results, chunk_size)

        print("\nModel: BM25")
        print("\nAccuracy for chunk_size of", chunk_size, "is ", acc_BM25)

        filename = "Data/csv_docs/accs_docs/accs_BM25_" + str(chunk_size) + ".csv"
        with open(filename, 'w', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(zip(k, acc_BM25))

    # Random ----------------------------------------------------------------------------------------------------------
        acc_random = evaluationRandom(n_results, text_chunks, chunk_size)

        print("\nModel: random")
        print("\nAccuracy for chunk_size of", chunk_size, "is ", acc_random)

        filename = "Data/csv_docs/accs_docs/accs_ramdom_" + str(chunk_size) + ".csv"
        with open(filename, 'w', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(zip(k, acc_random))

def eval_chunk_size(model_name, n_results, similarity_function):
    chunk_size_list = [2000, 4000, 8000]
    chunk_overlap_list = [200, 400, 800]
    accs = []
    for i in range(len(chunk_size_list)):
        chunk_size = chunk_size_list[i]
        chunk_overlap = chunk_overlap_list[i]
        text_chunks, ids = get_everything(chunk_size, chunk_overlap)
        get_chunks_id_csv_evaluation(text_chunks, ids, chunk_size)
        embedding = HuggingFaceEmbeddings(model_name = model_name)
        vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
        acc = evaluationVectorStores_chunk_size(vector_store, n_results, similarity_function, chunk_size)
        accs.append(acc)
        print("\nAccuracy for chunk_size of ", chunk_size, "is ", acc)

    print("\nAccuracy: ", accs)
    generate_plot_chunk_size(accs, n_results, chunk_size_list)

def eval_embeddings(n_results, similarity_function):

    chunk_size = 1000
    chunk_overlap = 100
    text_chunks, ids = get_everything(chunk_size, chunk_overlap)    

    # MiniLM ----------------------------------------------------------------------------------------------------------
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding = HuggingFaceEmbeddings(model_name = model_name)
    # vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
    vector_store = faiss_get_vector_store(embedding, model_name)
    accs_paraphrase_multilingual_MiniLM_L12_v2 = evaluationVectorStores(vector_store, n_results, similarity_function)

    # mpnet -----------------------------------------------------------------------------------------------------------
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    embedding = HuggingFaceEmbeddings(model_name = model_name)
    # vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
    vector_store = faiss_get_vector_store(embedding, model_name)
    accs_paraphrase_multilingual_mpnet_base_v2 = evaluationVectorStores(vector_store, n_results, similarity_function)

    # cased_v1 --------------------------------------------------------------------------------------------------------
    model_name = "distiluse-base-multilingual-cased-v1"
    embedding = HuggingFaceEmbeddings(model_name = model_name)
    # vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
    vector_store = faiss_get_vector_store(embedding, model_name)
    accs_distiluse_base_multilingual_cased_v1 = evaluationVectorStores(vector_store, n_results, similarity_function)

    # cased_v2 --------------------------------------------------------------------------------------------------------
    model_name = "distiluse-base-multilingual-cased-v2"
    embedding = HuggingFaceEmbeddings(model_name = model_name)
    # vector_store = faiss_embed_documents(embedding, text_chunks, model_name)
    vector_store = faiss_get_vector_store(embedding, model_name)
    accs_distiluse_base_multilingual_cased_v2 = evaluationVectorStores(vector_store, n_results, similarity_function)

    # BM25 ------------------------------------------------------------------------------------------------------------
    retriever = BM25Retriever.from_documents(text_chunks, k = n_results)
    accs_BM25 = evaluationBM25(retriever, n_results, chunk_size)

    # Random ----------------------------------------------------------------------------------------------------------
    accs_random = evaluationRandom(n_results, text_chunks, chunk_size)

    # Prints ----------------------------------------------------------------------------------------------------------
    print("\nminiLM\n")
    print(accs_paraphrase_multilingual_MiniLM_L12_v2)
    print("\nmpnet\n")
    print(accs_paraphrase_multilingual_mpnet_base_v2)
    print("\ncased_v1\n")
    print(accs_distiluse_base_multilingual_cased_v1)
    print("\ncased_v2\n")
    print(accs_distiluse_base_multilingual_cased_v2)
    print("\nbm25\n")
    print(accs_BM25)
    print("\nrandom\n")
    print(accs_random)

    # Plots -----------------------------------------------------------------------------------------------------------
    generate_plot_embeddings(accs_paraphrase_multilingual_MiniLM_L12_v2, accs_paraphrase_multilingual_mpnet_base_v2, accs_distiluse_base_multilingual_cased_v1, accs_distiluse_base_multilingual_cased_v2, accs_BM25, accs_random, n_results)

def get_chunk_position(result, correct_id):
    ids_result = []
    for element in result:
        ids_result.append(element.metadata["id"])
    # print(ids_result)
    if correct_id in(ids_result):
        return ids_result.index(correct_id) + 1
    else:
        return 0

def evaluationBM25(retriever, n_results, chunk_size):
    filename = "Data/csv_docs/questions_" + str(chunk_size) + ".csv"
    filename2 = "Data/csv_docs/paraphrases_" + str(chunk_size) + ".csv"
    
    data = pd.read_csv(filename2)
    data_dict = data.to_dict(orient="index")
    ids_result = []
    freq = [0] * n_results
    accs = [0] * n_results
    for i in range(len(data_dict)):
        result = retriever.invoke(data_dict[i]["questions"])
        id_result = get_chunk_position(result, data_dict[i]["ids"])
        if id_result != 0:
            for i in range(id_result - 1, n_results):
                freq[i] += 1
        ids_result.append(id_result)
    # print(ids_result)
    accs = [x / len(data_dict) for x in freq]
    return accs
        
def evaluationVectorStores(vector_store, n_results, similarity_function):
    data = pd.read_csv("Data/csv_docs/questions.csv")
    data_dict = data.to_dict(orient="index")
    ids_result = []
    freq = [0] * n_results
    accs = [0] * n_results
    for i in range(len(data_dict)):
        result = faiss_query(vector_store, data_dict[i]["questions"], n_results, similarity_function)
        id_result = get_chunk_position(result, data_dict[i]["ids"]) 
        if id_result != 0:
            for i in range(id_result - 1, n_results):
                freq[i] += 1
        ids_result.append(id_result)
    # print(ids_result)
    accs = [x / len(data_dict) for x in freq]
    return accs

def evaluationVectorStores_chunk_size(vector_store, n_results, similarity_function, chunk_size):
    filename = "Data/csv_docs/questions_" + str(chunk_size) + ".csv"
    filename2 = "Data/csv_docs/paraphrases_" + str(chunk_size) + ".csv"

    data = pd.read_csv(filename2)
    data_dict = data.to_dict(orient="index")
    ids_result = []
    freq = [0] * n_results
    accs = [0] * n_results
    for i in range(len(data_dict)):
        result = faiss_query(vector_store, data_dict[i]["questions"], n_results, similarity_function)
        id_result = get_chunk_position(result, data_dict[i]["ids"]) 
        if id_result != 0:
            for i in range(id_result - 1, n_results):
                freq[i] += 1
        ids_result.append(id_result)
    # print(ids_result)
    accs = [x / len(data_dict) for x in freq]
    return accs

def evaluationRandom(n_results, text_chunks, chunk_size):
    random.seed(5)
    filename = "Data/csv_docs/questions_" + str(chunk_size) + ".csv"
    filename2 = "Data/csv_docs/paraphrases_" + str(chunk_size) + ".csv"

    data = pd.read_csv(filename2)
    data_dict = data.to_dict(orient="index")
    ids = []
    for i in range(n_results):
        ids.append(random.randint(0, len(text_chunks)))
    ids_result = []
    freq = [0] * n_results
    accs = [0] * n_results
    for i in range(len(data_dict)):
        if data_dict[i]["ids"] in(ids):
            id_result = ids.index(data_dict[i]["ids"]) + 1
        else:
            id_result = 0
        if id_result != 0:
            for i in range(id_result - 1, n_results):
                freq[i] += 1
        ids_result.append(id_result)
    # print(ids_result)
    accs = [x / len(data_dict) for x in freq]
    return accs

def generate_plot_embeddings(accs_MiniLM, accs_mpnet, accs_cased_v1, accs_cased_v2, accs_BM25, accs_random, n_results):
    x_list = []
    for i in range(n_results):
        x_list.append(i)
    plt.plot(x_list, accs_MiniLM, label='MiniLM')
    plt.plot(x_list, accs_mpnet, label='mpnet')
    plt.plot(x_list, accs_cased_v1, label='cased_v1')
    plt.plot(x_list, accs_cased_v2, label='cased_v2')
    plt.plot(x_list, accs_BM25, label='BM25')
    plt.plot(x_list, accs_random, label='random')
    plt.xlabel("k")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def generate_plot_chunk_size(accs, n_results, chunk_size_list):
    x_list = []
    for i in range(n_results):
        x_list.append(i)
    for i in range(len(chunk_size_list)):
        plt.plot(x_list, accs[i], label=str(chunk_size_list[i]))

    plt.xlabel("k")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def getNumWords():
    chunk_size_list = [2000, 4000, 8000]
    chunk_overlap_list = [200, 400, 800]
    for i in range(len(chunk_size_list)):
        text_chunks, ids = get_everything(chunk_size_list[i], chunk_overlap_list[i])
        max = 0
        min = 10000
        avg = 0
        for chunk in text_chunks:
            tamanho = len(chunk.page_content.split())
            if tamanho > max:
                max = tamanho
            if tamanho < min:
                min = tamanho
            avg += tamanho
        print("Max of size " + str(chunk_size_list[i]) + " is " + str(max))
        print("Min of size " + str(chunk_size_list[i]) + " is " + str(min))
        print("Avg of size " + str(chunk_size_list[i]) + " is " + str(avg/len(text_chunks)))   

def getCorrectChunks(vector_store, chunk_size, chunk_overlap, n_results, similarity_function):
    filename = "Data/csv_docs/paraphrases_" + str(chunk_size) + ".csv"
    data = pd.read_csv(filename)
    data_dict = data.to_dict(orient="index")
    corretos_list = []
    perguntas_list = []

    for i in range(len(data_dict)):
        print(i, "\n")
        correto = 0
        query_text = data_dict[i]["questions"]
        correct_id = data_dict[i]["ids"]
        perguntas_list.append(query_text)
        result = faiss_query(vector_store, query_text, n_results, similarity_function)
        for element in result:
            if element.metadata['id'] == correct_id:
                correto = 1
        corretos_list.append(correto)
        
    filepath = "Data/csv_docs/chunks_correct/mpnet_"+ str(n_results) + ".csv"
    with open(filepath, 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(zip(corretos_list, perguntas_list))


def evaluateModels(vector_store, chunk_size, chunk_overlap, n_results, similarity_function):
    filename = "Data/csv_docs/paraphrases_" + str(chunk_size) + ".csv"
    data = pd.read_csv(filename)
    data_dict = data.to_dict(orient="index")

#  -- Maritalk -------------------------------------------------------------------------------------------------------------------

    # model = maritalk.MariTalk(
    #     key="111411717157153614044_940ada003417ac1a",
    #     model="sabia-2-medium"
    # )

    # filepath = "Data/csv_docs/model_docs/maritalk_"+ str(n_results) + ".csv"
    # responses = []
    # for i in range(len(data_dict)):
    #     print(i, "\n")
    #     documents_str = ""
    #     query_text = data_dict[i]["questions"]
    #     result = faiss_query(vector_store, query_text, n_results, similarity_function)
    #     for element in result:
    #         documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\md_docs\\", "")).replace(".md", "") + "\n"
    #         documents_str += "Conteúdo: " + element.page_content + "\n\n"
    #     response = generate_response_maritalk(model=model, documents_str=documents_str, query_text=query_text)
    #     responses.append(response)
    #     print(response)
    #     time.sleep(45)

    # with open(filepath, 'w', encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(responses))  

#  -- Gemini ------------------------------------------------------------------------------------------------------------------

    # genai.configure()
    # model = genai.GenerativeModel('gemini-1.5-pro')

    # filepath = "Data/csv_docs/model_docs/gemini-1.5-pro_"+ str(n_results) + ".csv"
    # responses = []
    # for i in range(len(data_dict)):
    #     print(i, "\n")
    #     documents_str = ""
    #     query_text = data_dict[i]["questions"]
    #     result = faiss_query(vector_store, query_text, n_results, similarity_function)
    #     for element in result:
    #         documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\md_docs\\", "")).replace(".md", "") + "\n"
    #         documents_str += "Conteúdo: " + element.page_content + "\n\n"
    #     response = generate_response_gemini(model, documents_str, query_text)
    #     responses.append(response)
    #     print(response)
    #     time.sleep(30)

    # with open(filepath, 'w', encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(responses))   

#  -- Claude ------------------------------------------------------------------------------------------------------------------

    # client = anthropic.Anthropic()
    # model_name = "claude-3-haiku-20240307"
    # max_tokens = 1024

    # filepath = "Data/csv_docs/model_docs/claude-3-5_" + str(n_results) + ".csv"
    # responses = []
    # for i in range(len(data_dict)):
    #     print(i, "\n")
    #     documents_str = ""
    #     query_text = data_dict[i]["questions"]
    #     result = faiss_query(vector_store, query_text, n_results, similarity_function)
    #     for element in result:
    #         documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\md_docs\\", "")).replace(".md", "") + "\n"
    #         documents_str += "Conteúdo: " + element.page_content + "\n\n"
    #     response = generate_response_claude(documents_str, query_text, model_name, max_tokens, client)
    #     responses.append(response)
    #     print(response)
    #     time.sleep(30)

    # with open(filepath, 'w', encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(responses))

#  -- Mixtral ------------------------------------------------------------------------------------------------------------------

    # model = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1")
    # filepath = "Data/csv_docs/model_docs/mistralai_"+ str(n_results) + ".csv"
    # responses = []
    # for i in range(len(data_dict)):
    #     print(i, "\n")
    #     documents_str = ""
    #     query_text = data_dict[i]["questions"]
    #     result = faiss_query(vector_store, query_text, n_results, similarity_function)
    #     for element in result:
    #         documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\md_docs\\", "")).replace(".md", "") + "\n"
    #         documents_str += "Conteúdo: " + element.page_content + "\n\n"
    #     response = generate_response_evaluation(model=model, documents_str=documents_str, query_text=query_text)
    #     responses.append(response)
    #     print(response)
    #     time.sleep(30)

    # with open(filepath, 'w', encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(responses))

#   -- Llama 3 ------------------------------------------------------------------------------------------------------------------

    model = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
    filepath = "Data/csv_docs/model_docs/llama3_" + str(n_results) + ".csv"
    responses = []
    for i in range(len(data_dict)):
        print(i, "\n")
        documents_str = ""
        query_text = data_dict[i]["questions"]
        result = faiss_query(vector_store, query_text, n_results, similarity_function)
        for element in result:
            documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\md_docs\\", "")).replace(".md", "") + "\n"
            documents_str += "Conteúdo: " + element.page_content + "\n\n"
        response = generate_response_evaluation(model=model, documents_str=documents_str, query_text=query_text)
        responses.append(response)
        print(response, "\n")
        time.sleep(30)

    with open(filepath, 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(zip(responses))

#   -- Gemma --------------------------------------------------------------------------------------------------------------------

#     model = HuggingFaceEndpoint(repo_id="google/gemma-7b")
#     filepath = "Data/csv_docs/model_docs/gemma7b_" + str(n_results) + ".csv"
#     responses = []
#     for i in range(len(data_dict)):
#         documents_str = ""
#         query_text = data_dict[i]["questions"]
#         result = faiss_query(vector_store, query_text, n_results, similarity_function)
#         for element in result:
#             documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\md_docs\\", "")).replace(".md", "") + "\n"
#             documents_str += "Conteúdo: " + element.page_content + "\n\n"
#         response = generate_response_evaluation(model=model, documents_str=documents_str, query_text=query_text)
#         responses.append(response)
#         print(response)

#     with open(filepath, 'w', encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerows(zip(responses))

#   -- GPT ----------------------------------------------------------------------------------------------------------------------

    # model = ChatOpenAI(model_name="gpt-3.5-turbo")
    # filepath = "Data/csv_docs/model_docs/gpt-3.5-turbo_" + str(n_results) + ".csv"
    # responses = []
    # for i in range(len(data_dict)):
    #     documents_str = ""
    #     query_text = data_dict[i]["questions"]
    #     result = faiss_query(vector_store, query_text, n_results, similarity_function)
    #     for element in result:
    #         documents_str += "Documento: " + ((element.metadata['source']).replace("Data\\md_docs\\", "")).replace(".md", "") + "\n"
    #         documents_str += "Conteúdo: " + element.page_content + "\n\n"
    #     response = generate_response_evaluation(model=model, documents_str=documents_str, query_text=query_text)
    #     responses.append(response.content)
    #     print(response.content)

    # with open(filepath, 'w', encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(responses))

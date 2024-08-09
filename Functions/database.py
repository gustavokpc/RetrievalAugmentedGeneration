import chromadb
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# CHROMA DB -------------------------------------------------------------------------------------------------------------------------------

def chroma_init():
    chroma_client = chromadb.PersistentClient()
    return chroma_client

def chroma_embed_documents(chroma_client, model_name, text_chunks, ids, similarity_function):
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    collection = chroma_client.get_or_create_collection(
        name="vectorstore",
        metadata={"hnsw:space": similarity_function},
        embedding_function=sentence_transformer_ef
        )

    collection.add(
        documents=text_chunks,
        ids = ids
    )
    return collection

def chroma_query(collection, query_texts_list, n_results):
    results = collection.query(
        query_texts=query_texts_list,
        n_results=n_results
    )
    return results

def use_chroma(text_chunks, similarity_function, n_results, ids, query_text_list):
    chroma_client = chroma_init()
    collection = chroma_embed_documents(chroma_client, "distiluse-base-multilingual-cased-v1", text_chunks, ids, similarity_function=similarity_function) # add similarity_function if needed
    results_chroma = chroma_query(collection, query_text_list, n_results)

    print(results_chroma)
    context_chroma = results_chroma["documents"]
    distances_chroma = results_chroma["distances"]

    print(context_chroma)
    print(distances_chroma)

    return context_chroma

# FAISS -----------------------------------------------------------------------------------------------------------------------------------

def faiss_embed_documents(embedding, text_chunks, model_name):
    faiss_db = FAISS.from_documents(text_chunks, embedding=embedding)
    if model_name == "paraphrase-multilingual-MiniLM-L12-v2":
        faiss_db.save_local("Databases/vector_store_paraphrase-multilingual-MiniLM-L12-v2")
    elif model_name == "paraphrase-multilingual-mpnet-base-v2":
        faiss_db.save_local("Databases/vector_store_paraphrase-multilingual-mpnet-base-v2")
    elif model_name == "distiluse-base-multilingual-cased-v1":
        faiss_db.save_local("Databases/vector_store_distiluse-base-multilingual-cased-v1")
    elif model_name == "distiluse-base-multilingual-cased-v2":
        faiss_db.save_local("Databases/vector_store_distiluse-base-multilingual-cased-v2")
    print("Created vector_store file path")
    return faiss_db

def faiss_get_vector_store(embedding, model_name):
    if model_name == "paraphrase-multilingual-MiniLM-L12-v2":
        vector_store = FAISS.load_local("Databases/vector_store_paraphrase-multilingual-MiniLM-L12-v2", embedding, allow_dangerous_deserialization=True)
    elif model_name == "paraphrase-multilingual-mpnet-base-v2":
        vector_store = FAISS.load_local("Databases/vector_store_paraphrase-multilingual-mpnet-base-v2", embedding, allow_dangerous_deserialization=True)
    elif model_name == "distiluse-base-multilingual-cased-v1":
        vector_store = FAISS.load_local("Databases/vector_store_distiluse-base-multilingual-cased-v1", embedding, allow_dangerous_deserialization=True)
    elif model_name == "distiluse-base-multilingual-cased-v2":
        vector_store = FAISS.load_local("Databases/vector_store_distiluse-base-multilingual-cased-v2", embedding, allow_dangerous_deserialization=True)
    print("Loaded vector_store file path")
    return vector_store

def faiss_query(vector_store, query_text, n_results, similarity_function):
    results_faiss = vector_store.similarity_search_with_score(query_text, k = n_results, collection_metadata={"hnsw:space": similarity_function})
    documents_faiss = []
    distances_faiss = []
    for documents in results_faiss:
        documents_faiss.append(documents[0])
        distances_faiss.append(documents[1])

    # print(distances_faiss)

    return documents_faiss

# def use_faiss(text_chunks, similarity_function, n_results, query_text):
#     embedding = HuggingFaceEmbeddings(model_name = "paraphrase-multilingual-MiniLM-L12-v2")
#     vector_store = faiss_embed_documents(embedding, text_chunks)
#     results_faiss = faiss_query(vector_store, query_text, n_results, similarity_function)

#     documents_faiss = []
#     distances_faiss = []

#     for documents in results_faiss:
#         documents_faiss.append(documents[0])
#         distances_faiss.append(documents[1])

#     print(distances_faiss)

#     return documents_faiss

# Parent Document Retriever TESTING -------------------------------------------------------------------------------------------------------

def parent_retriever(child_chunk_size, parent_chunk_size):
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=20)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=20)
    return child_splitter, parent_splitter

def create_retriever(vectorstore, child_splitter, parent_splitter):
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    return retriever

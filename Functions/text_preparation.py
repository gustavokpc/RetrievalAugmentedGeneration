from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import os
import csv

def get_text():

    # Get pdf texts
    # text = ""
    # directory = ""
    # metadatas = []
    # for pdf in testing_docs:
    #     pdf_reader = PdfReader(directory + pdf)
    #     for page in pdf_reader.pages:
    #         text += page.extract_text()
    #         metadatas.append(pdf)

    # Get markdown texts
    documents = DirectoryLoader("Data/md_docs", glob="*.md").load()

    return documents

def get_text_chunks(documents, chunk_size, chunk_overlap):

    # Divide text ("documents") into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        # add_start_index = True,
    )   
    chunks = text_splitter.split_documents(documents)
    print(f"Dividiu {len(documents)} documentos em {len(chunks)} chunks")
    return chunks

def get_text_ids(text_chunks):

    # Return list with the id for each chunk
    ids = []
    for id in range(len(text_chunks)):
        ids.append(id)
    return ids
    
def get_everything(chunk_size, chunk_overlap):

    # Return list of chunks and list of ids
    documents = get_text()
    text_chunks = get_text_chunks(documents, chunk_size, chunk_overlap)
    ids = get_text_ids(text_chunks)

    id = 0
    for chunk in text_chunks:
        chunk.metadata.update({"id": ids[id]})
        id += 1

    return text_chunks, ids

def get_chunks_id_csv(text_chunks, ids):

    # Create csv for text_chunks with the corresponding ids for a specified chunk_size
    with open("Data/csv_docs/chunks.csv", 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(zip(ids, text_chunks))

def get_chunks_id_csv_evaluation(text_chunks, ids, chunk_size):

    # Evaluation method that creates csv for text_chunks with the corresponding ids with indication of the chunk_size
    filename = "Data/csv_docs/chunks_eval_" + str(chunk_size) + ".csv"
    with open(filename, 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(zip(ids, text_chunks))

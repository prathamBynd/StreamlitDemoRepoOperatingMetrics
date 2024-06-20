import os
import shutil
import base64
import uuid
import re
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

from helperFunctions import download_blob_folder, load_from_pkl, retrieve_chunks, encode_image, pdf_to_images, extract_number, remove_currency, add_commas, convert_to_indian_system, highlight_text
from inferenceFunctions import get_response_haiku, get_type, metricDescription, filterChunks, choose_response


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

pinecone_api_key=os.getenv("PINECONE_API_KEY")

connection_string = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")


def get_answer(tickerName, companySector, input_metric, input_year):
    input_metric=input_metric.title()
    input_company=tickerName
    input_sector=companySector

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    try:
        blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports/{input_year}/{tickerName}_ar_{input_year}.pdf")
    except:
        return "Annual Report for this company is not present in our database."
    with open(f"{tickerName}_ar_{input_year}.pdf", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    input_pdf=f"{tickerName}_ar_{input_year}.pdf"

    blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports_Structured/{input_year}/TableChunks.json")
    with open("TableChunks.json", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports_Structured/{input_year}/TextChunks.json")
    with open("TextChunks.json", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports_Structured/{input_year}/UUIDDictionaries/tablePage_to_uuid.pkl")
    with open("tablePage_to_uuid.pkl", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports_Structured/{input_year}/UUIDDictionaries/textPage_to_uuid.pkl")
    with open("textPage_to_uuid.pkl", "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    download_blob_folder(f"{tickerName}/Annual_Reports_Structured/{input_year}/Vector Store Index/tableStorage", "tableStorage")
    download_blob_folder(f"{tickerName}/Annual_Reports_Structured/{input_year}/Vector Store Index/textStorage", "textStorage")


    with open('TableChunks.json', 'r') as file:
        data = json.load(file)
    tables = [item["text"] for item in data]
    table_page = [item["pageNo"] for item in data]
    with open('TextChunks.json', 'r') as file:
        data = json.load(file)
    text = [item["text"] for item in data]
    text_page = [item["pageNo"] for item in data]


    start_time = time.time()
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openai_api_key)

    pc = Pinecone(api_key=pinecone_api_key)

    pinecone_index_tables = pc.Index("tables")
    pinecone_index_text = pc.Index("text")
    vector_store_tables = PineconeVectorStore(
        pinecone_index=pinecone_index_tables,
        add_sparse_vector=True,
    )
    vector_store_text = PineconeVectorStore(
        pinecone_index=pinecone_index_text,
        add_sparse_vector=True,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to load pinecone: {elapsed_time} seconds")


    start_time = time.time()
    storage_context_tables = StorageContext.from_defaults(persist_dir="./tableStorage", vector_store=vector_store_tables)
    storage_context_text = StorageContext.from_defaults(persist_dir="./textStorage", vector_store=vector_store_text)
    tableIndex = load_index_from_storage(storage_context_tables)
    textIndex = load_index_from_storage(storage_context_text)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to load pinecone index: {elapsed_time} seconds")

    tablePage_to_uuid=load_from_pkl("./tablePage_to_uuid.pkl")
    textPage_to_uuid=load_from_pkl("./textPage_to_uuid.pkl")

    
    start_time = time.time()
    response=get_type(input_metric,input_company,input_sector)

    while True:
        try:
            ttype = json.loads(response)
            break
        except json.JSONDecodeError:
            print("Error: JSON decoding failed. Retrying...")
            ttype=get_type(input_metric,input_company,input_sector)

    ttype=ttype.get('Response')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to get type: {elapsed_time} seconds")


    if ttype=='QUANTITATIVE':
        start_time = time.time()
        description=metricDescription(input_metric, input_company)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to get description: {elapsed_time} seconds") 


        # start_time = time.time()
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="company", operator=FilterOperator.EQ, value=input_company
                ),
                MetadataFilter(
                    key="year", operator=FilterOperator.EQ, value=input_year
                ),
            ]
        )
        # retrieverTables = VectorIndexRetriever(index=tableIndex, similarity_top_k=10, sparse_top_k=7, query_mode="hybrid", filters=filters)
        # retrieverText = VectorIndexRetriever(index=textIndex, similarity_top_k=5, sparse_top_k=7, query_mode="hybrid", filters=filters)

        # retrievedNodesTables = retrieverTables.retrieve(f"{input_metric}")
        # retrievedNodesText = retrieverText.retrieve(f"{input_metric}")
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time to retrieve chunks: {elapsed_time} seconds")
        # print(f"Retrieved Chunks: {len(retrievedNodesTables) +len(retrievedNodesText)}")


        start_time = time.time()
        pages=set()
        chunkArgList=[(tableIndex, 10, input_metric, tablePage_to_uuid, filters), (textIndex, 5, input_metric, textPage_to_uuid, filters)]
        with ThreadPoolExecutor() as executor:
            results=executor.map(lambda args: retrieve_chunks(*args), chunkArgList)
        for x in results:
            for y in x:
                pages.add(y)
        pages=list(pages)
        pages.sort()
        pdf_to_images(input_pdf, pages)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to convert chunks to page images: {elapsed_time} seconds")

        start_time = time.time()
        inferences=[]
        responses=[]
        reasons=[]
        promptArgList=[]
        file_list = []
        print(pages)
        for filename in os.listdir("./jpegs"):
            if filename.endswith('.jpg'):
                file_list.append(filename)
        sorted_files = sorted(file_list, key=extract_number)

        for index, i in enumerate(sorted_files):
            print(i)
            context=[]
            image_path=os.path.join("./jpegs", i)

            for j,k in enumerate(table_page):
                if k==pages[index]:
                    context.append(tables[j])
            for j,k in enumerate(text_page):
                if k==pages[index]:
                    context.append(text[j])
            OCR = '\n\n'.join([f"{i+1}. {element}" for i, element in enumerate(context)])

            args=(input_metric, input_company, input_year, description, image_path, OCR)
            promptArgList.append(args)

        with ThreadPoolExecutor() as executor:
            results=executor.map(lambda args: filterChunks(*args), promptArgList)
            for inferencee in results:
                print(inferencee)
                while True:
                    try:
                        inference=json.loads(inferencee)
                        response=inference.get('Response')

                        if 'not' not in (str(response).lower()) and len(response)>0:
                            reason=inference.get('Reason')

                            inferences.append(inferencee)
                            responses.append(add_commas(remove_currency(response)))
                            reasons.append(reason)
                        else:
                            inferences.append(" ")

                        break
                    except json.JSONDecodeError:
                        print("Error: JSON decoding failed. Retrying...")
                        inferencee=filterChunks(input_metric, input_company, input_year, description, image_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to parallel process pages: {elapsed_time} seconds")

        start_time = time.time()
        finalInferences = []
        if len(responses)==0:
            finalAns="Metric not present in the report."
        elif len(set(responses))==1:
            finalAns=responses[0]

            for i in range(len(inferences)):
                if inferences[i]!=" ":
                    highlight_text(input_pdf, pages[i], pages, finalAns)
                    print(convert_to_indian_system(finalAns))
                    if not os.path.exists("highlighted_pdf.pdf"):
                        highlight_text(input_pdf, pages[i], pages, convert_to_indian_system(finalAns))
                    break
        else:
            inferences_numbered_list = "\n\n".join(f"{i+1}. {inference}" for i, inference in enumerate(inferences) if inference!=" ")
            answerr=choose_response(input_metric, input_company, input_year, description, inferences_numbered_list)
            print(answerr)
        
            while True:
                try:
                    answer=json.loads(answerr)

                    finalAns=answer.get('Response')
                    highlightText=add_commas(remove_currency(finalAns))
                    reason=answer.get('Reason')
                    page_number=pages[int(remove_currency(str(answer.get('Point Number'))))-1]
                    print(page_number)

                    break
                except json.JSONDecodeError:
                    print("Error: JSON decoding failed. Retrying...")
                    answer=choose_response(input_metric, input_company, input_year, description, inferences_numbered_list)

            for i in inferences:
                if i !=" ":
                    if answerr not in i:
                        finalInferences.append(i)
            print(highlightText)
            highlight_text(input_pdf, page_number, pages, highlightText)
            if not os.path.exists("highlighted_pdf.pdf"):
                highlight_text(input_pdf, page_number, pages, convert_to_indian_system(highlightText))

        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to get final answer and highlighting: {elapsed_time} seconds")


    if os.path.exists("./jpegs"):
        shutil.rmtree("./jpegs")


    return finalAns, finalInferences


if __name__ == "__main__":
    finalAns, finalInferences=get_answer("ABB", "engineering", "number of employees", "2022")

    print(finalAns)
    print(finalInferences)
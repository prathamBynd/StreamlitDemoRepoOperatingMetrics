import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
import anthropic
import shutil
import base64
import uuid
import os
import re
import json
import pickle
import fitz 
from pdf2image import convert_from_path

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

from openai import OpenAI

from dotenv import load_dotenv

import time

from concurrent.futures import ThreadPoolExecutor

from azure.storage.blob import BlobServiceClient


load_dotenv()


anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
client=anthropic.Anthropic(api_key=anthropic_api_key)
haiku = "claude-3-haiku-20240307"

openai_api_key = os.getenv("OPENAI_API_KEY")
oclient=OpenAI(api_key=openai_api_key)

pinecone_api_key=os.getenv("PINECONE_API_KEY")

connection_string = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")


def download_blob_folder(blob_folder_path, local_folder_name, connection_string=connection_string, container_name=container_name):
    # Create a BlobServiceClient object using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get a container client
    container_client = blob_service_client.get_container_client(container_name)

    # List blobs in the specified folder
    blob_list = container_client.list_blobs(name_starts_with=blob_folder_path)
    
    if not os.path.exists(local_folder_name):
        os.makedirs(local_folder_name)

    # Download each blob
    for blob in blob_list:
        # Get the blob name relative to the folder
        relative_path = blob.name[len(blob_folder_path) + 1:]
        # Create the local path
        local_path = os.path.join(local_folder_name, relative_path)
        
        # Create local directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Get a blob client for the blob you want to download
        blob_client = container_client.get_blob_client(blob.name)
        
        # Download the blob to a local file
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())


def load_from_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def categorize_elements(raw_pdf_elements):
    text_elements=[]
    text_data = []
    table_elements = []
    table_data=[]
    for element in raw_pdf_elements:
        if 'CompositeElement' in str(type(element)):
            text_elements.append(element)
            text_data.append(str(element))
        elif 'Table' in str(type(element)):
            table_elements.append(element)
            table_data.append(str(element))
    return text_elements, text_data, table_elements, table_data


def retrieve_chunks(indexName, retrievalNumber, input_metric, dictionaryType):
    retrieverTables = VectorIndexRetriever(index=indexName, similarity_top_k=retrievalNumber, sparse_top_k=7, query_mode="hybrid")

    retrievedNodesTables = retrieverTables.retrieve(f"{input_metric}")

    pages = {dictionaryType[node.node.id_] for node in retrievedNodesTables if node.node.id_ in dictionaryType}

    return pages


def get_response_haiku(message):
    response = client.messages.create(
        model=haiku,
        max_tokens=1024,
        messages=message
    )
    return response.content[0].text


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_type(input_metric,input_company,input_sector):
    messages=[
    {
        "role": "user",
        "content": [
        {"type": "text", "text": f'''You are provided with the following user query whose answer can be found in the annual report of a publicly listed company:

        User Query: {input_metric}
        Company Name: {input_company}
        Sector of the Company: {input_sector}
        
        Based on the query, tell whether the user is asking for a QUALITATIVE answer or a QUANTITATIVE answer.
        
        Return only 'QUALITATIVE' or 'QUANTITATIVE' in your answer with no extra text or information.
        
        Give your response in the following JSON schema:-

        {{
            "Response": ""
        }}'''}
        ],
    }
    ]

    response=get_response_haiku(messages)

    return response


def pdf_to_images(pdf_path, pagesList):

    os.makedirs("./jpegs", exist_ok=True)

    for i in pagesList:
        pages = convert_from_path(pdf_path, first_page=i,
        last_page=i,fmt='jpeg', output_file='page', paths_only=True,
        output_folder="./jpegs", dpi=400)


def is_variable_defined(var_name):
    return var_name in globals() or var_name in locals()


def metricDescription(input_metric, input_company):
    response=oclient.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":  f'''Give me a concise but useful explanation of the following metric - "{input_metric}". Include some common ways in which this might be mentioned as well. This description is going to be used to search for the metric in an annual report for "{input_company}" company.
                        
Give me the output in JSON format with following fields - 
{{
metric_description: "Description",
common_mentions: "list of common mentions",
search_instruction_string: "list of search instruction strings"
}}'''
                    },
                ],
            }
        ],
            max_tokens=300
    )

    return response.choices[0].message.content


def extract_number(filename):
    return int(filename.split('-')[1].split('.')[0])


def filterChunks(input_metric, input_company, input_year, description, image_path, OCR):
    response=oclient.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url":  {
                            "type": "base64",
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                        }
                    },
                    {
                        "type": "text",
                        "text":  f'''You have been provided a METRIC, Description of METRIC and a COMPANY name below-
                        
METRIC: {input_metric}
Description of METRIC: {description}
COMPANY: {input_company}

Give me numerical information about the METRIC for the COMPANY corresponding to the year {input_year} on the basis of the given images of tables from the company's annual report along with the Description of METRIC given.

I have also performed OCR on the relevant section of the page so you can refer the below text to know the correct text in case unclear in the image-

Text from relevant sections in the page: {OCR}
                    
Only give me the numeric information about what is asked and do not return any extra text or information in your response. Give preference to concrete numbers rather than percentages. Make sure your answer is a value that is mentioned in one of the tables and also includes the complete unit and denomination of the value.

If the answer is not present in the images, do not make any assumptions or guesses and return 'METRIC NOT PRESENT' in your reponse.

Return your answer in the following JSON format-

{{
"Response": "Numerical value answer or METRIC NOT PRESENT",
"Reason": "Reason why you thought this number was the answer by mentioning along with the table, column it was present in along with the heading given to the table telling what it is about"
}}'''
                    },
                ],
            }
        ],
            max_tokens=300
    )

    return response.choices[0].message.content

# What is the value of for the following metric for {company name}?

# Metric Name -
def choose_response(input_metric, input_company, input_year, description, inferences):
    print(inferences)
    response=oclient.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":  f'''A financial analyst has been provided a METRIC, Description of METRIC and a COMPANY name below-
                        
METRIC: {input_metric}
Description of METRIC: {description}
COMPANY: {input_company}

The analyst was to go through the COMPANY's annual report and give the numerical value corresponding to the METRIC for the year {input_year} along with the reason why they think the value they chose in the report was the correct one.

Here are a bunch of responses point-wise that the analyst gave consisting of the values they chose along with their reasons-

{inferences}

Return the one you think has the most logical reason and is likely to be the correct one considering the table and column it was present in. Just return the exact one you think is the correct one in the following JSON format-
{{
    "Response": "The exact response which has the correct answer",
    "Point Number": "The point number corresponding to that response in the list of responses given above"
}}'''
                    },
                ],
            }
        ],
            max_tokens=300
    )

    return response.choices[0].message.content


def remove_currency(text):
    # Define regular expression pattern to match currency symbols and text
    pattern = r'[^\d.]'  # Match any character that is not a digit or comma

    # Remove currency symbols and text
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text


def add_commas(number_str):
    try:
        # Check if the input string contains a decimal point
        if '.' in number_str:
            # Convert to float if it has a decimal
            num = float(number_str)
        else:
            # Convert to int if it doesn't have a decimal
            num = int(number_str)
        
        # Format the number with commas and return as a string
        return f"{num:,}"
    except ValueError:
        # Handle the case where the input cannot be converted to a float or int
        raise ValueError("Input must be a number that can be converted to a float or int")


def highlight_text(pdf_path, page_number, pages, text_to_highlight):
    pages.insert(0,page_number)

    for page_number in pages:
        # Open the PDF file
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Check if the page number is valid
        if page_number < 1 or page_number > len(pdf_reader.pages):
            print("Invalid page number.")
            return

        # Open the PDF using PyMuPDF
        doc = fitz.open(pdf_path)

        # Get the page
        page = doc.load_page(page_number - 1)

        # Search for the matched text and highlight it
        text_instances = page.search_for(f" {text_to_highlight} ")

        if len(text_instances)==0:
            text_instances = page.search_for(f" {text_to_highlight}")
            if len(text_instances)==0:
                text_instances = page.search_for(f"{text_to_highlight} ")
                if len(text_instances)==0:
                    text_instances = page.search_for(f"{text_to_highlight}")
                    if len(text_instances)==0:
                        continue

        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)

        for i in range(len(doc) - 1, -1, -1):
            if i != page_number - 1:
                doc.delete_page(i)

        # Save the modified PDF with highlights
        output_pdf_path = './highlighted_pdf.pdf'
        doc.save(output_pdf_path)
        doc.close()

        # Close the PDF file
        pdf_file.close()

        break


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

    pinecone_index_tables = pc.Index(f"{tickerName.lower()}-tables-{input_year}")
    pinecone_index_text = pc.Index(f"{tickerName.lower()}-text-{input_year}")
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


        start_time = time.time()
        retrieverTables = VectorIndexRetriever(index=tableIndex, similarity_top_k=10, sparse_top_k=7, query_mode="hybrid")
        retrieverText = VectorIndexRetriever(index=textIndex, similarity_top_k=5, sparse_top_k=7, query_mode="hybrid")

        retrievedNodesTables = retrieverTables.retrieve(f"{input_metric}")
        retrievedNodesText = retrieverText.retrieve(f"{input_metric}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to retrieve chunks: {elapsed_time} seconds")


        start_time = time.time()
        pages=set()
        chunkArgList=[(tableIndex, 10, input_metric, tablePage_to_uuid), (textIndex, 5, input_metric, textPage_to_uuid)]
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
        if len(responses)==0:
            finalAns="Metric not present in the report."
        elif len(set(responses))==1:
            finalAns=responses[0]

            for i in range(len(inferences)):
                if inferences[i]!=" ":
                    highlight_text(input_pdf, pages[i], pages, finalAns)
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

            print(highlightText)
            highlight_text(input_pdf, page_number, pages, highlightText)

        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to get final answer and highlighting: {elapsed_time} seconds")


    if os.path.exists("./jpegs"):
        shutil.rmtree("./jpegs")

    finalInferences = []
    for i in inferences:
        if i !=" ":
            if answerr not in i:
                finalInferences.append(i)


    return finalAns, finalInferences


if __name__ == "__main__":
    print(get_answer("AADHARHousinggg", "Finance service commercial", "salaried borrowers", "2023"))
from unstructured.partition.pdf import partition_pdf
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from azure.storage.blob import BlobServiceClient
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import json
import uuid
import pickle
import shutil
import os


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
oclient=OpenAI(api_key=openai_api_key)

pinecone_api_key=os.getenv("PINECONE_API_KEY")


def split_pdf(pdf_path, output_folder="./splitPDF"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        
        for page_number in range(len(pdf_reader.pages)):
            pdf_writer = PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[page_number])
            
            output_file_path = os.path.join(output_folder, f"page_{page_number + 1}.pdf")
            
            with open(output_file_path, 'wb') as output_file:
                pdf_writer.write(output_file)


def numerical_sort(value):
    return int(value.split('_')[-1].split('.')[0])


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
    return text_elements, table_elements


def append_to_json(file_path, text, page_no):
    # Read the existing JSON file if it exists
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    except json.JSONDecodeError:
        data = []

    # Append the new data
    new_entry = {
        "text": text,
        "pageNo": page_no
    }
    data.append(new_entry)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def upload_folder_to_blob(container_client, local_folder_path, blob_folder_path):
    for root, dirs, files in os.walk(local_folder_path):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            # Construct the path in the blob container
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            blob_path = os.path.join(blob_folder_path, relative_path).replace("\\", "/")

            # Get a blob client
            blob_client = container_client.get_blob_client(blob_path)
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data)


def load_from_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_to_pkl(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def generate_uuid():
    return str(uuid.uuid4())


def chunkReport(reportPath, tickerName, year):
    if os.path.exists("highlighted_pdf.pdf"):
        os.remove("highlighted_pdf.pdf")
        
    connection_string = os.getenv("AZURE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_CONTAINER_NAME")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    container_client = blob_service_client.get_container_client(container_name)
    
    split_pdf(reportPath)
    
    pdf_elements=[]
    for pageNo, j in enumerate(sorted(os.listdir("splitPDF"), key=numerical_sort)):
        try:
            pdf_elements_temp = partition_pdf(
                f"splitPDF/{j}",
                chunking_strategy="by_title",
                infer_table_structure=True,
                max_characters=3000,
                new_after_n_chars=2800,
                combine_text_under_n_chars=2000
            )

        except Exception as e:
            print(f"Error processing page {pageNo + 1}: {e}")
            continue

        for element in pdf_elements_temp:
            try:
                element.metadata.orig_elements[0].metadata.page_number = pageNo+ 1
            except AttributeError as e:
                print(f"Error setting page number for element: {e}")
                continue

        pdf_elements+=pdf_elements_temp
        print(f"Chunked page: {pageNo + 1}")

    texts, tables = categorize_elements(pdf_elements)


    for i in tables:
        append_to_json("TableChunks.json", str(i).split(" Identity Number")[0], i.metadata.orig_elements[0].metadata.page_number)
    for i in texts:
        append_to_json("TextChunks.json", str(i).split(" Identity Number")[0], i.metadata.orig_elements[0].metadata.page_number)

    with open('TableChunks.json', 'r') as file:
        data = json.load(file)
    table_info = [item["text"] for item in data]
    table_page = [item["pageNo"] for item in data]
    with open('TextChunks.json', 'r') as file:
        data = json.load(file)
    text_info = [item["text"] for item in data]
    text_page = [item["pageNo"] for item in data]

    pc = Pinecone(api_key=pinecone_api_key)

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openai_api_key)
    
    # pc.create_index(
    #     name="text",
    #     dimension=3072,
    #     metric="dotproduct",
    #     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    # )
    # pc.create_index(
    #     name=f"tables",
    #     dimension=3072,
    #     metric="dotproduct",
    #     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    # )

    pinecone_index_text = pc.Index("text")
    pinecone_index_tables = pc.Index("tables")

    vector_store_text = PineconeVectorStore(
        pinecone_index=pinecone_index_text,
        add_sparse_vector=True,
    )
    vector_store_tables = PineconeVectorStore(
        pinecone_index=pinecone_index_tables,
        add_sparse_vector=True,
    )

    storage_context_text = StorageContext.from_defaults(vector_store=vector_store_text)
    storage_context_tables = StorageContext.from_defaults(vector_store=vector_store_tables)

    textNodes=[]
    textPage_to_uuid = {}
    for index, i in enumerate(text_info):
        if len(i)>0:
            node_id = generate_uuid()
            textPage_to_uuid[node_id]=text_page[index]
            textNodes.append(TextNode(text=i, id_=node_id, metadata={"company": tickerName, "year": year}))
            
    save_to_pkl(textPage_to_uuid,"textPage_to_uuid.pkl")
    textIndex=VectorStoreIndex(textNodes, storage_context=storage_context_text)
    textIndex.storage_context.persist(persist_dir="textStorage")

    tableNodes=[]
    tablePage_to_uuid = {}
    for index, i in enumerate(table_info):
        if len(i)>0:
            node_id = generate_uuid()
            tablePage_to_uuid[node_id]=table_page[index]
            tableNodes.append(TextNode(text=i, id_=node_id, metadata={"company": tickerName, "year": year}))
            
    save_to_pkl(tablePage_to_uuid,"tablePage_to_uuid.pkl")
    tableIndex=VectorStoreIndex(tableNodes, storage_context=storage_context_tables)
    tableIndex.storage_context.persist(persist_dir="tableStorage")


    blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports_Structured/{year}/TableChunks.json")
    with open("TableChunks.json", "rb") as data:
        blob_client.upload_blob(data)
    blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports_Structured/{year}/TextChunks.json")
    with open("TextChunks.json", "rb") as data:
        blob_client.upload_blob(data)

    blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports_Structured/{year}/UUIDDictionaries/tablePage_to_uuid.pkl")
    with open("tablePage_to_uuid.pkl", "rb") as data:
        blob_client.upload_blob(data)
    blob_client = container_client.get_blob_client(f"{tickerName}/Annual_Reports_Structured/{year}/UUIDDictionaries/textPage_to_uuid.pkl")
    with open("textPage_to_uuid.pkl", "rb") as data:
        blob_client.upload_blob(data)

    upload_folder_to_blob(container_client, "tableStorage", f"{tickerName}/Annual_Reports_Structured/{year}/Vector Store Index/tableStorage")
    upload_folder_to_blob(container_client, "textStorage", f"{tickerName}/Annual_Reports_Structured/{year}/Vector Store Index/textStorage")

    shutil.rmtree("splitPDF")

if __name__ == "__main__":
    chunkReport("./reports/TCS_ar_2023.pdf", "TCS", "2023")
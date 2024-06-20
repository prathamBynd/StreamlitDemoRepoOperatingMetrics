import os
import pickle
import base64

import anthropic
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from llama_index.core.retrievers import VectorIndexRetriever
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
import fitz
from pdf2image import convert_from_path


load_dotenv()

connection_string = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME")


def load_from_pkl(filepath):
    """
    Loads a Python object from a pickle (.pkl) file.

    Args:
    - filepath (str): Path to the pickle file to be loaded.

    Returns:
    - Loaded Python object: The object loaded from the pickle file.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
    
def download_blob_folder(blob_folder_path, local_folder_name, connection_string, container_name):
    """
    Downloads all blobs from a specified folder in Azure Blob Storage to a local directory.

    Args:
    - blob_folder_path (str): Path of the folder inside the Azure Blob Storage container.
    - local_folder_name (str): Local directory where blobs will be downloaded.
    - connection_string (str): Azure Storage account connection string.
    - container_name (str): Name of the Azure Blob Storage container.

    Returns:
    None
    """
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs(name_starts_with=blob_folder_path)
    
    if not os.path.exists(local_folder_name):
        os.makedirs(local_folder_name)

    for blob in blob_list:
        relative_path = blob.name[len(blob_folder_path) + 1:]
        local_path = os.path.join(local_folder_name, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        blob_client = container_client.get_blob_client(blob.name)
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())


def retrieve_chunks(indexName, retrievalNumber, input_metric, dictionaryType, filters):
    """
    Retrieves chunks of data from a vector index based on a specified metric and returns associated pages.

    Args:
    - indexName (str): Name of the vector index to retrieve data from.
    - retrievalNumber (int): Number of results to retrieve.
    - input_metric (str): Metric used for retrieval.
    - dictionaryType (dict): Dictionary mapping node IDs to pages.

    Returns:
    - set: Set of pages associated with retrieved nodes.
    """
    retrieverTables = VectorIndexRetriever(index=indexName, similarity_top_k=retrievalNumber, sparse_top_k=7, query_mode="hybrid", filters=filters)
    retrievedNodesTables = retrieverTables.retrieve(input_metric)

    pages = {dictionaryType[node.node.id_] for node in retrievedNodesTables if node.node.id_ in dictionaryType}

    return pages


def encode_image(image_path):
    """
    Encodes an image file to Base64 format.

    Args:
    - image_path (str): Path to the image file to be encoded.

    Returns:
    - str: Base64 encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def pdf_to_images(pdf_path, pagesList):
    """
    Converts specific pages of a PDF file into JPEG images and saves them to a directory.

    Args:
    - pdf_path (str): Path to the PDF file that needs to be converted.
    - pagesList (list): List of page numbers (1-indexed) to convert into images.

    Returns:
    - None
    """
    os.makedirs("./jpegs", exist_ok=True)

    for i in pagesList:
        convert_from_path(pdf_path, first_page=i, last_page=i,
                          fmt='jpeg', output_file=f'page_{i}', paths_only=True,
                          output_folder="./jpegs", dpi=400)


def extract_number(filename):
    """
    Extracts a number from a filename formatted as 'prefix-number.extension'.

    Args:
    - filename (str): The filename from which to extract the number.

    Returns:
    - int: The extracted number.

    """
    return int(filename.split('-')[1].split('.')[0])


def remove_currency(text):
    """
    Removes currency symbols and non-numeric characters (except periods) from a given text string.

    Args:
    - text (str): The input text that may contain currency symbols and other non-numeric characters.

    Returns:
    - str: A cleaned text string containing only digits and periods.

    """
    pattern = r'[^\d.]'  # Match any character that is not a digit or period
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def add_commas(number_str):
    """
    Converts a string representation of a number into a formatted string with commas for better readability.

    Args:
    - number_str (str): A string representing a number, which can be an integer or a float.

    Returns:
    - str: The input number string formatted with commas.

    Raises:
    - ValueError: If the input cannot be converted to a float or int.

    """
    try:
        if '.' in number_str:
            num = float(number_str)
        else:
            num = int(number_str)
        
        return f"{num:,}"
    
    except ValueError:
        raise ValueError("Input must be a number that can be converted to a float or int")


def convert_to_indian_system(number_str):
    """
    Converts a string representation of a number into the Indian numbering system format.

    The Indian numbering system formats large numbers by placing commas after every two digits from the rightmost digit,
    starting from the least significant digit of the integer part.

    Args:
    - number_str (str): A string representing a number, which can be an integer or a float.

    Returns:
    - str: The input number string formatted in the Indian numbering system format.

    """
    if '.' in number_str:
        integer_part, decimal_part = number_str.split('.')
    else:
        integer_part, decimal_part = number_str, None

    integer_part = integer_part.replace(',', '')

    reversed_integer = integer_part[::-1]

    indian_reversed = ''
    for i in range(len(reversed_integer)):
        if i > 2 and (i - 1) % 2 == 0:
            indian_reversed += ','
        indian_reversed += reversed_integer[i]

    indian_integer = indian_reversed[::-1]

    if decimal_part:
        result = indian_integer + '.' + decimal_part
    else:
        result = indian_integer

    return result


def highlight_text(pdf_path, page_number, pages, text_to_highlight):
    """
    Highlights a specific text in a PDF document and saves the modified PDF with highlighted text.

    Args:
    - pdf_path (str): Path to the input PDF file.
    - page_number (int): Page number where the text should be highlighted.
    - pages (list): List of additional page numbers to process.
    - text_to_highlight (str): Text string to be highlighted in the PDF.

    Returns:
    - None

    """
    pages.insert(0, page_number)

    for page_number in pages:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        if page_number < 1 or page_number > len(pdf_reader.pages):
            print("Invalid page number.")
            return

        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)

        text_instances = page.search_for(f" {text_to_highlight} ")

        if len(text_instances) == 0:
            text_instances = page.search_for(f" {text_to_highlight}")
            if len(text_instances) == 0:
                text_instances = page.search_for(f"{text_to_highlight} ")
                if len(text_instances) == 0:
                    text_instances = page.search_for(f"{text_to_highlight}")
                    if len(text_instances) == 0:
                        continue

        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)

        for i in range(len(doc) - 1, -1, -1):
            if i != page_number - 1:
                doc.delete_page(i)

        output_pdf_path = './highlighted_pdf.pdf'
        doc.save(output_pdf_path)
        doc.close()

        pdf_file.close()

        break
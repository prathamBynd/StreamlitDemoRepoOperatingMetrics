import streamlit as st 
from streamlit_pdf_viewer import pdf_viewer

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

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode

from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore

from openai import OpenAI

from dotenv import load_dotenv

import time


load_dotenv()


anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
client=anthropic.Anthropic(api_key=anthropic_api_key)
haiku = "claude-3-haiku-20240307"
sonnet = "claude-3-sonnet-20240229"

openai_api_key = os.getenv("OPENAI_API_KEY")
oclient=OpenAI(api_key=openai_api_key)

pinecone_api_key=os.getenv("PINECONE_API_KEY")


companyList=["Aadhar Housing Finance"]
input_sector="Finance service commercial"
pdfList=["./AadharHousingFinanceAnnualReport.pdf"]


def split_pdf(pdf_path, output_folder="./splitPDF"):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        
        # Iterate through each page
        for page_number in range(len(pdf_reader.pages)):
            # Create a PdfWriter object for each page
            pdf_writer = PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[page_number])
            
            # Output file name for the page
            output_file_path = os.path.join(output_folder, f"page_{page_number + 1}.pdf")
            
            # Write the page to the output file
            with open(output_file_path, 'wb') as output_file:
                pdf_writer.write(output_file)


def numerical_sort(value):
    return int(value.split('_')[-1].split('.')[0])


def generate_uuid():
    return str(uuid.uuid4())


def save_to_pkl(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

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


def get_response_haiku(message):
    response = client.messages.create(
        model=haiku,
        max_tokens=1024,
        messages=message
    )
    return response.content[0].text


def get_response_sonnet(message):
    response = client.messages.create(
        model=sonnet,
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
        output_folder="./jpegs", dpi=600)


def is_variable_defined(var_name):
    return var_name in globals() or var_name in locals()


def metricDescription(input_metric, input_company):
    response=oclient.chat.completions.create(
        model="gpt-4o",
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


def filterChunks(input_metric, input_company, input_year, description, image_path):
    response=oclient.chat.completions.create(
        model="gpt-4o",
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

Give me numerical information about the METRIC for the company mentioned in the year {input_year} on the basis of the given images of tables from the company's annual report along with the Description of METRIC given.
                    
Only give me the numeric information about what is asked and do not return any extra text or information in your response. Give preference to concrete numbers rather than percentages. Make sure your answer is a value that is mentioned in one of the tables and also includes the complete unit and denomination of the value.

If the answer is not present in the images, do not make any assumptions or guesses and return 'METRIC NOT PRESENT' in your reponse.

Return your answer as a simple text that looks like JSON without a ```json prefix in the following format-

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


def choose_response(input_metric, input_company, input_year, description, inferences):
    print(inferences)
    response=oclient.chat.completions.create(
        model="gpt-4o",
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

Return the one you think has the most logical reason and is likely to be the correct one considering the table and column it was present in. Do not have any extra text, information or formatting in your response - just return the exact one you think is the correct one. Do not add ```json prefix, I want normal text that looks like this JSON-
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

        # Extract the text from the specified page
        page_text = pdf_reader.pages[page_number - 1].extract_text()

        # Open the PDF using PyMuPDF
        doc = fitz.open(pdf_path)

        # Get the page
        page = doc.load_page(page_number - 1)

        page_text = page.get_text()

        # Search for the matched text and highlight it
        text_instances = page.search_for(text_to_highlight)

        if len(text_instances)==0:
            continue 

        valid_instances = []

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


def main():
    st.title("Bynd's Aadhar Housing Finance Metric Extractor")

    # Prompt the user to enter a metric
    input_metric = st.text_input("Ask for any metric from AHF's 10-K 2023 Report:")
    input_year = st.text_input("Enter the year corresponding to the metric:")

    # Run the function if a metric is entered
    if input_metric and input_year:
        results=[]
        source=""

        for i in range(len(pdfList)):
            input_company=companyList[i]
            input_pdf=pdfList[i]


            pdf_elements=[]

            input_company_pickle_filename = f"./{input_company}.pkl"

            if not os.path.exists(input_company_pickle_filename):
                continue

            else:
                pdf_elements = load_from_pkl(input_company_pickle_filename)


            texts_elements, texts, tables, tables_text = categorize_elements(pdf_elements)
            
            start_time = time.time()
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openai_api_key)

            pc = Pinecone(api_key=pinecone_api_key)

            pinecone_index_tables = pc.Index("tables")
            vector_store_tables = PineconeVectorStore(
                pinecone_index=pinecone_index_tables,
                add_sparse_vector=True,
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time to load pinecone: {elapsed_time} seconds")


            if os.path.exists("./tableStorage") and os.path.exists("./tablePage_to_uuid.pkl"):
                if not is_variable_defined('tableIndex'):
                    
                    start_time = time.time()
                    storage_context_tables = StorageContext.from_defaults(persist_dir="./tableStorage", vector_store=vector_store_tables)
                    tableIndex = load_index_from_storage(storage_context_tables)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time to load pinecone index: {elapsed_time} seconds")

                # if not is_variable_defined('textIndex'):
                #     storage_context_text = StorageContext.from_defaults(persist_dir="./textStorage")
                #     textIndex = load_index_from_storage(storage_context_text)

                tablePage_to_uuid=load_from_pkl("./tablePage_to_uuid.pkl")
                # textPage_to_uuid=load_from_pkl("./textPage_to_uuid.pkl")


            else:
                tableNodes = []
                tablePage_to_uuid = {}

                pc.create_index(
                    name="tables",
                    dimension=3072,
                    metric="dotproduct",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                pinecone_index_tables = pc.Index("tables")
                vector_store_tables = PineconeVectorStore(
                    pinecone_index=pinecone_index_tables,
                    add_sparse_vector=True,
                )
                
                for index, i in enumerate(tables_text):
                    if len(i) > 0:
                        node_id = generate_uuid()
                        tablePage_to_uuid[node_id] = tables[index].metadata.orig_elements[0].metadata.page_number
                        tableNodes.append(TextNode(text=i, id_=node_id))
                
                save_to_pkl(tablePage_to_uuid, "./tablePage_to_uuid.pkl")
                storage_context_tables = StorageContext.from_defaults(vector_store=vector_store_tables)
                tableIndex = VectorStoreIndex(tableNodes, storage_context=storage_context_tables)
                tableIndex.storage_context.persist(persist_dir="./tableStorage")
                

                # textNodes=[]
                # textPage_to_uuid = {}
                # for index, i in enumerate(texts):
                #     if len(i)>0:
                #         node_id = generate_uuid()
                #         textPage_to_uuid[node_id]=texts_elements[index].metadata.orig_elements[0].metadata.page_number
                #         textNodes.append(TextNode(text=i, id_=node_id))
                        
                # save_to_pkl(textPage_to_uuid,"./textPage_to_uuid.pkl")
                # storage_context_text = StorageContext.from_defaults(vector_store=vector_store_text)
                # textIndex=VectorStoreIndex(textNodes, storage_context=storage_context_text)
                # textIndex.storage_context.persist(persist_dir="./textStorage")
            

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
                retriever = VectorIndexRetriever(index=tableIndex, similarity_top_k=10, sparse_top_k=7, query_mode="hybrid")

                retrievedNodes = retriever.retrieve(f"{input_metric}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Time to retrieve chunks: {elapsed_time} seconds")


                start_time = time.time()
                pages = {tablePage_to_uuid[node.node.id_] for node in retrievedNodes if node.node.id_ in tablePage_to_uuid}
                pages=list(pages)
                pages.sort()
                pdf_to_images(input_pdf, pages)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Time to convert chunks to page images: {elapsed_time} seconds")


                inferences=[]
                responses=[]
                reasons=[]
                for i in os.listdir("./jpegs"):
                    print(i)
                    image_path=os.path.join("./jpegs", i)

                    inferencee=filterChunks(input_metric, input_company, input_year, description, image_path)
                    print(inferencee)

                    while True:
                        try:
                            inference=json.loads(inferencee)
                            response=inference.get('Response')

                            if 'not' not in response.lower() and len(response)>0:
                                reason=inference.get('Reason')

                                inferences.append(inferencee)
                                responses.append(add_commas(remove_currency(response)))
                                reasons.append(reason)
                            else:
                                inferences.append(" ")

                            break
                        except json.JSONDecodeError:
                            print("Error: JSON decoding failed. Retrying...")
                            inference=filterChunks(input_metric, input_company, input_year, description, image_path)


                if len(responses)==0:
                    st.write("Metric not present in the report.")
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
                
                source="table"


            if len(responses)>0:
                st.write(finalAns)
                if os.path.exists("./highlighted_pdf.pdf"):
                    pdf_viewer("./highlighted_pdf.pdf")

            if os.path.exists("./jpegs"):
                shutil.rmtree("./jpegs")
            if os.path.exists("./highlighted_pdf.pdf"):
                shutil.rmtree("./highlighted_pdf.pdf")

if __name__ == "__main__":
    main()
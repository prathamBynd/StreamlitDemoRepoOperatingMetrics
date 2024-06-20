import os
from openai import OpenAI
import anthropic
from dotenv import load_dotenv


load_dotenv()

anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
client=anthropic.Anthropic(api_key=anthropic_api_key)
haiku = "claude-3-haiku-20240307"

openai_api_key = os.getenv("OPENAI_API_KEY")
oclient=OpenAI(api_key=openai_api_key)


def get_response_haiku(message):
    """
    Generates a haiku response using an API client and returns the generated text.

    Args:
    - message (str): Input message or prompt to generate the haiku response.

    Returns:
    - str: Generated haiku text response.
    """
    response = client.messages.create(
        model=haiku,
        max_tokens=1024,
        messages=message
    )
    return response.content[0].text


def get_type(input_metric, input_company, input_sector):
    """
    Determines whether a user query regarding a metric of a publicly listed company
    is asking for a qualitative or quantitative answer.

    Args:
    - input_metric (str): The specific metric or question from the user.
    - input_company (str): The name of the company the query pertains to.
    - input_sector (str): The sector to which the company belongs.

    Returns:
    - dict: A JSON-like dictionary with the response indicating whether the user query
      is 'QUALITATIVE' or 'QUANTITATIVE'.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": f'''You are provided with the following user query whose answer can be found in the annual report of a publicly listed company:

                    User Query: {input_metric}
                    Company Name: {input_company}
                    Sector of the Company: {input_sector}
                    
                    Based on the query, tell whether the user is asking for a QUALITATIVE answer or a QUANTITATIVE answer.
                    
                    Return only 'QUALITATIVE' or 'QUANTITATIVE' in your answer with no extra text or information.
                    
                    Give your response in the following JSON schema:-
                    
                    {{
                        "Response": ""
                    }}'''
                }
            ],
        }
    ]

    response = get_response_haiku(messages)

    return response


def metricDescription(input_metric, input_company):
    """
    Requests a concise but informative description of a metric, along with common mentions and search instructions,
    using OpenAI's GPT-4o model.

    Args:
    - input_metric (str): The metric for which a description is requested.
    - input_company (str): The company's name for which the metric description is tailored.

    Returns:
    - str: A JSON-formatted string containing the metric description, common mentions, and search instructions.
    """
    response = oclient.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'''Give me a concise but useful explanation of the following metric - "{input_metric}". Include some common ways in which this might be mentioned as well. This description is going to be used to search for the metric in an annual report for "{input_company}" company.

Give me the output in JSON format with the following fields -
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


def filterChunks(input_metric, input_company, input_year, description, image_path, OCR):
    """
    Requests numerical information about a metric from images of tables in a company's annual report,
    based on a provided metric description and OCR text.

    Args:
    - input_metric (str): The metric for which numerical information is requested.
    - input_company (str): The name of the company for which the metric information is requested.
    - input_year (str or int): The year corresponding to which the metric information is requested.
    - description (str): Description of the metric provided for context.
    - image_path (str): Path to the image file containing tables from the annual report.
    - OCR (str): Text extracted via OCR from relevant sections of the page for clarification.

    Returns:
    - str: JSON-formatted response containing the numerical value of the metric or 'METRIC NOT PRESENT'
           if the metric information is not found in the images.

    """
    response = oclient.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "type": "base64",
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                        }
                    },
                    {
                        "type": "text",
                        "text": f'''You have been provided a METRIC, Description of METRIC and a COMPANY name below-

METRIC: {input_metric}
Description of METRIC: {description}
COMPANY: {input_company}

Give me numerical information about the METRIC for the COMPANY corresponding to the year {input_year} on the basis of the given images of tables from the company's annual report along with the Description of METRIC given.

I have also performed OCR on the relevant section of the page so you can refer the below text to know the correct text in case unclear in the image-

Text from relevant sections in the page: {OCR}

Only give me the numeric information about what is asked and do not return any extra text or information in your response. Give preference to concrete numbers rather than percentages. Make sure your answer is a value that is mentioned in one of the tables and also includes the complete unit and denomination of the value.

If the answer is not present in the images, do not make any assumptions or guesses and return 'METRIC NOT PRESENT' in your response.

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
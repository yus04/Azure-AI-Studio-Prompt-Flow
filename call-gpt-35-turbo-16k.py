from promptflow import tool
from promptflow.connections import CustomConnection
from openai import AzureOpenAI
import json

def get_aoai_client(azure_endpoint: str, api_key: str, api_version: str) -> AzureOpenAI:
    client = AzureOpenAI(
        azure_endpoint = azure_endpoint, 
        api_key = api_key,  
        api_version = api_version
    )
    return client

def chat_completion(client: AzureOpenAI, model: str, message: str) -> str:
    response = client.chat.completions.create(
        model = model,
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

@tool
def call_gpt(message: str, myconn: CustomConnection) -> str:
    AZURE_OPENAI_ENDPOINT = myconn.AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY = myconn.AZURE_OPENAI_API_KEY
    api_version = "2024-03-01-preview"
    model = "gpt-35-turbo-16k"
    client = get_aoai_client(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, api_version)
    content = chat_completion(client, model, message)
    return content

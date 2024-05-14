# message の例：東京の天気は雪ですか
# 1106 以降のモデルでないと、複数の関数を返してくれない

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

def get_current_weather(location: str, unit : str = "fahrenheit") -> str:
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def check_snow(temperature: str) -> bool:
    """Check if the temperature indicates snow"""
    int_temperature = int(temperature)
    snow = int_temperature <= 32
    return snow

def get_tools() -> [dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_snow",
                "description": "Check if the temperature indicates snow",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "temperature": {
                            "type": "number",
                            "description": "The temperature in Fahrenheit"
                        }
                    },
                    "required": ["temperature"]
                },
            },
        }
    ]
    return tools

def function_calling(client: AzureOpenAI, model: str, messages: str, tools: [dict]) -> (dict, [str]):
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        tools = tools
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    messages.append(response_message)
    return messages, tool_calls

def call_functions(client: AzureOpenAI, messages: [dict], tool_calls: [str], model: str) -> [dict]:
    if not tool_calls: return None
    available_functions = {
        "get_current_weather": get_current_weather,
        "check_snow": check_snow,
    }
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        if function_to_call == get_current_weather:
            function_response = function_to_call(
                location = function_args.get("location"),
                unit = function_args.get("unit"),
            )
        elif function_to_call == check_snow:
            function_response = function_to_call(
                temperature = function_args.get("temperature")
            )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )
    response_message = client.chat.completions.create(
        model = model,
        messages = messages
    )
    return response_message

@tool
def call_gpt(message: str, myconn: CustomConnection) -> str:
    AZURE_OPENAI_ENDPOINT = myconn.AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY = myconn.AZURE_OPENAI_API_KEY
    api_version = "2024-03-01-preview"
    model = "gpt-35-turbo-16k"
    messages = [{"role": "user", "content": message}]
    client = get_aoai_client(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, api_version)
    tools = get_tools()
    messages, tool_calls = function_calling(client, model, messages, tools)
    response_message = call_functions(client, messages, tool_calls, model)
    response_message_content = response_message.choices[0].message.content
    return response_message_content

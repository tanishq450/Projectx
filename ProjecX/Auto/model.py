from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient

def get_model():
    return OpenAIChatCompletionClient(model="gemini-2.5-flash",
                                          api_key="AIzaSyACbX4GBw2SFtHslGBeSlrmsP2O6pd_kJ0",
                                          model_info={
                                                "max_tokens": 4096, 
                                                "vision":True,
                                                "temperature": 0.1,
                                                "top_p": 0.9,
                                                "top_k": 40,
                                                "presence_penalty": 0.0,
                                                "function_calling": "auto",
                                                "json_output":"true",
                                                "family":"google"
                                          }
    )
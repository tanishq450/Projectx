from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient

def get_model():
    return OllamaChatCompletionClient(model="gpt-oss:120b-cloud",
                                          model_info={
                                                "max_tokens": 4096, 
                                                "vision":True,
                                                "temperature": 0.1,
                                                "top_p": 0.9,
                                                "top_k": 40,
                                                "presence_penalty": 0.0,
                                                "function_calling": "auto",
                                                "json_output":"true",
                                                "family":"gpt"
                                          }
    )
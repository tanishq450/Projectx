from autogen_agentchat.agents import AssistantAgent
import tavily
from model import get_model
from langchain_tavily import TavilySearch
from loguru import logger
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
import asyncio
from prompt import web_search_prompt



logger=logger.bind(name="web_search_agent")

def web_search(query: str):
    try:
        tavily_search_tool = TavilySearch(
            tavily_api_key="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",
            max_results=1,
            top_k=1,
        )
        result = tavily_search_tool.run(query)
        logger.info("Web search completed successfully")
        return result
    except Exception as e:
        logger.error(e)
        return None

    
def web_search_agent():
    agent = AssistantAgent(
        name="websearch_agent",
        model_client=get_model(),
        tools=[web_search],
        system_message=web_search_prompt
    )
    return agent


"""if __name__ == "__main__":
    agent = web_search_agent()
    response = asyncio.run(agent.run(task="Who won world test championship 2025"))
    print(response.messages[-1].content)"""




from autogen_agentchat.agents import AssistantAgent
from model import get_model
from loguru import logger
import asyncio
##from prompt import rag_prompt
from Llama_index.Rag_pipeline import Rag_pipeline

logger= logger.bind(name="rag_agent")

rag=Rag_pipeline()


def run_tool(query: str):
    try:
        logger.info("Running RAG pipeline")
        response=rag.run_pipeline(
            "/home/tanishq/ProjecX/Llama_index/1706.03762v7.pdf",
            query
        )
        logger.info("RAG pipeline completed successfully")
        return response
    except Exception as e:
        logger.error(e)
        return None


def rag_agent():
    return AssistantAgent(
        name="rag_agent",
        model_client=get_model(),
        system_message=" you are a rag agent that can answer questions based on the documents provided to you and alwways use tool for answering",
        tools=[run_tool],

    )



if __name__ == "__main__":
    agent = rag_agent()
    response = asyncio.run(agent.run(task="what is attention mechanism"))
    print(response.messages[-1].content)

    


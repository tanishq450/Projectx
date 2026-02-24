from autogen_agentchat.agents import AssistantAgent
from model import get_model
from loguru import logger
import asyncio
from prompt import validator_prompt

logger= logger.bind(name="validator_agent")



def validator_agent():
    return AssistantAgent(
        name="validator_agent",
        model_client=get_model(),
        system_message=validator_prompt,
    )

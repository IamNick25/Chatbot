import openai
import langchain
import pydantic
import typing
import os
import termcolor
from termcolor import colored
from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, Tool
from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)



class CalculatorInput(BaseModel):
    question: str = Field()

class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "useful for when you need to answer questions about current events"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return llm_math_chain.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")

def main():
    print(colored('Welcome to the chatbot', 'green', attrs=['bold']))
    print(colored('You can start chatting with the bot.', 'green', attrs=['bold']))

    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [
        CustomSearchTool(),
        CustomCalculatorTool()
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

    while True:
        user_input = input(colored('You: ', 'yellow'))

        if user_input.lower() == 'exit':
            break

        try:
            response = agent.run(user_input)
            print(colored('Bot: ', 'green', attrs=['bold']) + response)

        except Exception as error:
            print(colored(str(error), 'red'))

if __name__ == "__main__":
    main()

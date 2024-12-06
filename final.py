import openai
import langchain
import pydantic
import typing
import os
from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.callbacks.manager import (
  AsyncCallbackManagerForToolRun,
  CallbackManagerForToolRun,
)
import termcolor
from termcolor import colored

def main():
  print(colored('Welcome to the chatbot', 'green', attrs=['bold']))
  print(colored('You can start chatting with the bot.', 'green', attrs=['bold']))

chat_history = []

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools = [
  Tool.from_function(
    func=search.run,
    name="Search",
    description=
    "useful for when you need to answer questions about current events"
     
  ),
]


class CalculatorInput(BaseModel):
 question: str = Field()


tools.append(
  Tool.from_function(
    func=llm_math_chain.run,
    name="Calculator",
    description="useful for when you need to answer questions about math",
    args_schema=CalculatorInput
     
  ))
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

agent = initialize_agent(tools,
             llm,
             agent=AgentType.OPENAI_FUNCTIONS,
             verbose=True)

try:
  while True:
    user_input = input(colored('You: ', 'yellow'))
    messages = [{'role': role, 'content': content} for role, content in chat_history]
    messages.append({'role': 'user', 'content': user_input})
    
    chat_completion = agent.run(messages)
    
    print(colored('Bot: ', 'green', attrs=['bold']) + chat_completion)
    
    chat_history.append(('user', user_input))
    chat_history.append(('assistant', chat_completion))

except Exception as error:
print(colored(str(error), 'red'))

if _name_ == "_main_":
  main()
import openai
import langchain
import pydantic
import typing
import os
from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
#from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import termcolor
from termcolor import colored
from langchain.agents import load_tools
#from pydantic import BaseModel, validator, Field
from typing import Optional, Any, Type
from langchain.tools.base import Tool
from pydantic.v1 import BaseModel, Field
import flask
from flask import Flask, request, jsonify



tools = load_tools(["serpapi"])


def main():
  print(colored('Welcome to the chatbot', 'green', attrs=['bold']))
  print(
      colored('You can start chatting with the bot.', 'green', attrs=['bold']))


chat_history = []

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm, verbose=True)
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description=
        "useful for when you need to answer questions about current events"),
]

#class CalculatorInput(BaseModel):
#question: str = Field()


class CalculatorInput(BaseModel):
  my_base_model_subclass: Type[BaseModel] = Field(
      ...,
      description=
      "Equivalent to the args_schema field in langchain/StructuredTool")


my_foo = CalculatorInput(my_base_model_subclass=CalculatorInput)

print(f"My foo {my_foo} is successfully instantiated")

#class CustomCalculatorTool(BaseTool):
#name = "Calculator"
#description = "useful for when you need to answer questions about math"
#args_schema: type[BaseModel] = Foo  # Ensure it's a Type[BaseModel]
tools.append(
    Tool.from_function(  # <-- tool uses v1 namespace
        func=lambda question: 'hello',
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput))

#tools.append(
#Tool.from_function(
# func=llm_math_chain.run,
#name="Calculator",
# description="useful for when you need to answer questions about math",
#args_schema=Foo))#

#@validator("args_schema")
#def validate_args_schema(cls, val):
#if issubclass(type(val), CalculatorInput):
#return val

#raise TypeError(
#"Wrong type for 'args_schema', must be subclass of CalculatorInput")


def _run(self,
         query: str,
         run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
  """Use the tool."""
  return llm_math_chain.run(query)


async def _arun(
    self,
    query: str,
    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
  """Use the tool asynchronously."""
  raise NotImplementedError("Calculator does not support async")


agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.OPENAI_FUNCTIONS,
                         verbose=True)

try:
  while True:
       app = flask.Flask(__name__)



       @app.route('/')
       def index():
          return flask.send_file('index.html')

@app.route('/send-message', methods=['POST'])
def send_message():
  user_input = request.json['message']
  messages = [{'role': role, 'content': content} for role, content in chat_history]
  messages.append({'role': 'user', 'content': user_input})

  chat_completion = agent.run(messages)

  print(colored('Bot: ', 'green', attrs=['bold']) + chat_completion)

  chat_history.append(('user', user_input))
  chat_history.append(('assistant', chat_completion))


except Exception as error:
print(colored(str(error), 'red'))




if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
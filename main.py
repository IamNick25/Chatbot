import flask
from flask import Flask, request, jsonify, render_template
import openai
import langchain
from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
import termcolor
from termcolor import colored
from langchain.agents import load_tools, AgentType, initialize_agent
import pydantic
from pydantic.v1 import BaseModel, Field
import flask_cors

import cors
from flask_cors import CORS
import path
import json
import sympy
from sympy import symbols, integrate
from sympy.parsing.sympy_parser import parse_expr
import re

app = Flask(__name__)
CORS(app)

tools = load_tools(["serpapi"])

chat_history = []

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm, verbose=True)

tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description=
        "useful for when you need to answer questions about current events")]


'''class CalculatorInput(BaseModel):
  my_base_model_subclass: Type[BaseModel] = Field(
      ...,
      description=
      "Equivalent to the args_schema field in langchain/StructuredTool")


my_foo = CalculatorInput(my_base_model_subclass=CalculatorInput)

tools.append(
    Tool.from_function(
        func=lambda question: 'hello',
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput))'''
def extract_expression(user_input):
    """
    Extracts the mathematical expression from a user input string.
    Assumes the user input is in a format like "integrate x^2" or "What is the integral of x**3".
    """
    # Using regular expression to find a mathematical expression within the input
    # This is a  and mbasic exampleight need to be adjusted based on actual input formats
    match = re.search(r'integrate (.+)', user_input, re.IGNORECASE)
    if match:
        expression = match.group(1)
        # Replace any '^' with '**' for sympy compatibility
        expression = expression.replace('^', '**')
        return expression
    return None
def perform_integration(expression_str):
  x = symbols('x')  # Define the symbol to use in the expression
  try:
      expression = parse_expr(expression_str)  # Parse the expression string
      return integrate(expression, x)  # Perform integration
  except Exception as e:
      return str(e)
class CalculatorInput(BaseModel):
 question: str = Field()


tools.append(
  Tool.from_function(
    func=llm_math_chain.run,
    name="Calculator",
    description="useful for when you need to answer questions about math",
    args_schema=CalculatorInput

  ))

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.OPENAI_FUNCTIONS,
                         verbose=True)


@app.route('/')
def index():
  return "sucess"


@app.route('/send-message', methods=['POST'])
def send_message():
  user_input = request.json['message']
  '''if "integrate" in user_input.lower():
    # Extract the expression from the user input here
    # This extraction depends on how you expect the input to be formatted
    # For example, if the input is "integrate x^2", you extract "x**2"
    expression = extract_expression(user_input)  # You'll need to implement this
    result = perform_integration(expression)
    return jsonify({'message': result})'''
  if "integrate" in user_input.lower():
    expression = extract_expression(user_input)
    if expression:
        result = perform_integration(expression)
        return jsonify({'message': str(result)})
    else:
        return jsonify({'message': 'Could not extract expression for integration.'})
  print("Received user input:", user_input)
  message = [{
      'role': role,
      'content': content
  } for role, content in chat_history]
  message.append({'role': 'user', 'content': user_input})

  chat_completion = agent.run(message)

  #print(colored('Bot: ', 'green', attrs=['bold']) + chat_completion)

  chat_history.append(('user', user_input))
  chat_history.append(('assistant', chat_completion))

  return jsonify({'message': chat_completion})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8005)
  print("server is running on 8005 port")

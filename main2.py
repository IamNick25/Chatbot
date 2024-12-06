import openai
import langchain
import os
from langchain import OpenAI, SerpAPIWrapper, LLMMathChain
from langchain.agents import AgentType, initialize_agent, load_tools, Tool

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

tools =[
    Tool([name="Search",
         func=search.run,
         description=
         "useful for when you need to answer questions about current events."]),load_tools(["wikipedia", "wolframalpha"], llm=llm)]

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.OPENAI_FUNCTIONS,
                         verbose=True)

val = input("You: ")

res = agent.run(val)
print(res)
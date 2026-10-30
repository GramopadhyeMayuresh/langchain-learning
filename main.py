from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


llm = ChatOpenAI(model_name="gpt-5",temperature=0.5)
tools = [TavilySearch()]
agent = create_agent(llm, tools)

def main():
    response = agent.invoke({"messages":[HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Pune(India) or remote area on linkedin and list their details?")]});
    print(response);

if __name__ == "__main__":
    main()

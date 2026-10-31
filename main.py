from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url: str = Field(description="The URL of the source")
    title: str = Field(default="", description="The title or description of the source")


class JobPosting(BaseModel):
    """
    Schema for an individual job posting
    """
    title: str = Field(description="The job title")
    company: str = Field(description="The company name")
    location: str = Field(description="The job location (e.g., Pune, Remote)")
    description: str = Field(description="Key requirements and highlights of the role")
    url: str | None = Field(default=None, description="Direct URL to the job listing if available")


class AgentResponse(BaseModel):
    """
    Schema for the agent's structured response including job details and sources
    """
    answer: str = Field(description="Summary or direct answer to the user query")
    jobs: List[JobPosting] = Field(default_factory=list, description="List of job postings found")
    sources: List[Source] = Field(default_factory=list, description="List of sources and links used")



llm = ChatOpenAI(model_name="gpt-5", temperature=0.5)
tools = [TavilySearch()]
agent = create_agent(llm, tools, response_format=AgentResponse)

def main():
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for an ai engineer using langchain in the Pune(India) or remote area on linkedin and list their details?"
                )
            ]
        }
    )
    
    structured_response: AgentResponse = response.get("structured_response")
    if structured_response:
        print("=== Answer ===")
        print(structured_response.answer)
        
        if structured_response.jobs:
            print("\n=== Job Postings ===")
            for i, job in enumerate(structured_response.jobs, 1):
                print(f"\n{i}. {job.title} - {job.company} ({job.location})")
                print(f"   Description: {job.description}")
                if job.url:
                    print(f"   Link: {job.url}")

        if structured_response.sources:
            print("\n=== Sources ===")
            for source in structured_response.sources:
                print(f"- {source.title}: {source.url}" if source.title else f"- {source.url}")
    else:
        print(response)

if __name__ == "__main__":
    main()

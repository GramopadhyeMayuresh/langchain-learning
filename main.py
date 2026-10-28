from importlib import machinery
from importlib import machinery
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello World")
    information = """
    Elon Reeve Musk is a business magnate and investor. He is the founder, CEO, and chief technology officer of SpaceX; angel investor, CEO, and product architect of Tesla, Inc.; owner and CTO of X Corp., and founder of The Boring Company and co-founder of Neuralink and OpenAI.    
    Born = June 28, 1971 (age 55), Pretoria, South Africa
    Nationality = South African, Canadian, American
    Spouse = Justine Musk, Talulah Riley
    Children = X Æ A-12, Griffin Musk, Vivian Jenna Wilson, Kai Musk, Saxon Musk, Damus "
    "+Musk, X Æ A-Xii Musk


    """

    prompt = PromptTemplate(
        input_variables=["information", "question"],
        template="""
        Use the following information to answer the question:
        {information}

        Question: {question}
        """,
    )
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOllama(model="gemma3:4b", temperature=0)
    chain = prompt | llm
    response = chain.invoke({"information": information, "question": "Who is Elon Musk?"})
    print(response.content)



if __name__ == "__main__":
    main()

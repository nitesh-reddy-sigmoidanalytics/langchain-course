from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
    Narendra Damodardas Modi (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 2014.
    """

    summary_template = """
    Given the information {information} about a person, create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})

    print(response.content)


if __name__ == "__main__":
    main()

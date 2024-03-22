import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="Python")
args = parser.parse_args()

api_key = os.getenv("OPENAI_API_KEY")


llm = OpenAI(openai_api_key=api_key)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a unit test for the following {language} code:\n{code}",
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"],
)

result = chain({"language": args.language, "task": args.task})

print(">>>>>>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>>>>>> GENERATED TEST:")
print(result["test"])

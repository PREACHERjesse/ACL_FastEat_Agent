from dotenv import load_dotenv
from langchain.agents import create_agent
from .prompt import SYSTEM_CHEF_PROMPT,CHEF_PROMPT
from langchain.messages import HumanMessage
from .tools import web_search
from langchain_openai import ChatOpenAI
from .photo_decoder import choose_image_file,build_image_message
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

load_dotenv()


model = ChatOpenAI(
    model="gpt-5.6-sol",
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_BASE_URL"),
    use_responses_api=False,
)
logging.info("creating out agent...")
agent = create_agent(
    model = model,
    tools = [web_search],
    system_prompt =SYSTEM_CHEF_PROMPT
)

logging.info("choosing our image...")
path = choose_image_file()
message = build_image_message(path,CHEF_PROMPT)

logging.info("invoking our agent...")
response = agent.invoke(
    {"messages":[message]}
)

print(response["messages"][-1].content)

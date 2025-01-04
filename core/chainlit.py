import os

import chainlit as cl
from dotenv import load_dotenv

from core.pipeline import RecommendationPipeline

load_dotenv()
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

pipeline = RecommendationPipeline(
    neo4j_uri=neo4j_uri,
    neo4j_username=neo4j_username,
    neo4j_password=neo4j_password,
)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        "Hello! I'm your hotel booking assistant. How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    response = pipeline.run(message.content)
    await cl.Message(content=response).send()



@cl.on_stop
async def on_stop():
    await cl.Message(
        "Thank you for using the hotel booking assistant."
        "If you need help in the future, "
        "feel free to reach out!"
    ).send()


@cl.on_chat_end
async def on_chat_end():
    await cl.Message("Goodbye!").send()

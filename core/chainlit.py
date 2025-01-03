import os

import chainlit as cl
from dotenv import load_dotenv

from core.pipeline import GraphRAG


@cl.on_message
async def main(message: cl.Message):
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    graph_rag = GraphRAG(
        openai_api_key=openai_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
    )

    response = await graph_rag.execute_pipeline(message.content)
    await cl.Message(content=response).send()

import asyncio
import json
import os
from typing import Any, Dict, List

import community.community_louvain as community_louvain
import networkx as nx
import openai
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from core.agent import FuncationCallingAgent
from core.graphdb import HotelRecommender


class GraphRAG:
    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: str = None,
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
    ):
        """
        Initialize GraphRAG with LLM-driven functionality.
        """
        openai.api_key = openai_api_key
        self.openai_model = openai_model
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password

    async def call_process_recommendations(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to parse a user query and extract parameters based on predefined lists.

        Args:
            query (str): Natural language query.

        Returns:
            Dict[str, Any]: Extracted parameters.
        """
        self.agent = FuncationCallingAgent(
            llm=OpenAI(self.openai_model),
            tools=[FunctionTool.from_defaults(self.process_recommendations)],
            timeout=120,
            verbose=True,
        )

        agent_response = await self.agent.run(input=query)

        return agent_response["tool_output"]

    def process_recommendations(
        self, amenities: List[str], stay_type: str, stay_duration: str
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Process recommended_hotels data and convert it into element instances.

        Params:
            - amenities: A list of required amenities from this list: ["Air Conditioning", "TV", "Balcony", "Food Service", "Parking", "Vehicle Hire"]
            - stay_duration: One of ["Short", "Medium", "Long"]
            - stay_type: One of ["Couple", "Family", "Group", "Solo traveller"]

        Example params: (just for reference)
            amenities = ["TV", "Parking"]
            stay_type = "Family"
            stay_duration = "Long"

        Returns:
            List[Dict[str, Any]]: Structured element instances.
        """
        recommender = HotelRecommender(
            self.neo4j_uri, self.neo4j_username, self.neo4j_password
        )

        recommended_hotels = recommender.recommend_hotels(
            amenities=amenities, stay_type=stay_type, stay_duration=stay_duration
        )

        recommender.close()
        return recommended_hotels

    async def execute_pipeline(self, query: str):
        """
        Full pipeline execution starting from recommendations to insights.

        Args:
            query (str): Natural language query for the pipeline.
        """
        print("Parsing query...")
        instances = await self.call_process_recommendations(query)
        print(instances)


if __name__ == "__main__":
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

    query = "Find hotels with TV and Parking for a long family stay."

    asyncio.run(graph_rag.execute_pipeline(query))

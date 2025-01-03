import asyncio
import json
import os
from typing import Any, Dict, List

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

    def global_answer(self, community_answers: List[Dict[str, Any]], query: str) -> str:
        """
        Step 2: Global answer
        Synthesize community-level answers into a comprehensive response.

        Args:
            community_answers (List[Dict[str, Any]]): List of summaries or answers for each community.
            query (str): The global query to be answered based on community-level information.

        Returns:
          str: The final synthesized global answer.
        """
        try:
            final_response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Synthesize the following intermediate answers into a single, comprehensive response to the global query.
                        The final response must:
                        - Be concise yet thorough.
                        - Address the query directly.
                        - Integrate all relevant information from the intermediate answers.
                        - Eliminate redundancy and ensure logical coherence.
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\nIntermediate Answers: {community_answers}",
                    },
                ],
            )

            return final_response.choices[0].message.content

        except Exception as e:
            print(f"Error generating global answer: {e}")
            return "Unable to generate a global answer at this time."

    async def execute_pipeline(self, query: str):
        """
        Full pipeline execution starting from recommendations to insights.

        Args:
            query (str): Natural language query for the pipeline.
        """
        print("Parsing query...")
        community_answers = await self.call_process_recommendations(query)

        output = self.global_answer(community_answers=community_answers, query=query)
        return output


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

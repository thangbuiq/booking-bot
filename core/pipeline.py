import asyncio
import os
from typing import Any
from typing import Dict
from typing import List

import openai
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from neo4j import GraphDatabase

from core.agent import FuncationCallingAgent
from core.graphdb import HotelRecommender


class RecommendationPipeline:
    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
    ):
        """
        Initialize GraphRAG with LLM-driven functionality.
        """
        self.openai_model = openai_model
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password

    async def cypher_recommendations(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to parse a user query and extract parameters based on predefined lists.

        Args:
            query (str): Natural language query.

        Returns:
            Dict[str, Any]: Extracted parameters.
        """
        self.agent = FuncationCallingAgent(
            llm=OpenAI(self.openai_model),
            tools=[FunctionTool.from_defaults(self.process_cypher_recommendations)],
            timeout=120,
            verbose=True,
        )

        agent_response = await self.agent.run(input=query)

        return agent_response["tool_output"]

    def process_cypher_recommendations(
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

    def process_graph_rag_recommendations(self, query: str) -> List[Dict[str, Any]]:
        communities = self.communities_detection()
        communities_summaries = self.communities_summarization(communities)
        recommendation_hotels = self.global_answer(communities_summaries, query)

        return recommendation_hotels

    def hybrid_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """
        Step 1: Hybrid recommendations
        Generate recommendations based on a hybrid approach combining LLM and graph-based methods.

        Args:
            query (str): Natural language query.

        Returns:
            List[Dict[str, Any]]: List of recommended hotels.
        """
        cypher_recommendations = self.cypher_recommendations(query)
        graph_rag_recommendations = self.process_graph_rag_recommendations(query)

        # Rerank

        return cypher_recommendations + graph_rag_recommendations

    def communities_detection(self):
        """
        Detect communities within the graph database using Neo4j's community detection algorithms.

        Returns:
            List[Dict[str, Any]]: List of detected communities with their respective nodes.
        """

        driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password)
        )
        with driver.session() as session:
            query = """
            CALL algo.louvain.stream()
            YIELD nodeId, community
            RETURN algo.asNode(nodeId).name AS node, community
            ORDER BY community
            """
            result = session.run(query)
            communities = {}
            for record in result:
                community = record["community"]
                node = record["node"]
                if community not in communities:
                    communities[community] = []
                communities[community].append(node)

        driver.close()
        return [{"community": k, "nodes": v} for k, v in communities.items()]

    def communities_summarization(
        self, communities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Step 2: Community summarization
        Generate summaries for each community based on the nodes within the community.

        Args:
            communities (List[Dict[str, Any]]): List of detected communities with their respective nodes.

        Returns:
            List[Dict[str, Any]]: List of summaries for each community.
        """
        summaries = []
        for community in communities:
            nodes = community["nodes"]
            summary = f"Community {community['community']} includes the following nodes: {nodes}"
            summaries.append({"community": community["community"], "summary": summary})
        return summaries

    def local_answer(self, community: Dict[str, Any], query: str) -> str:
        """
        Step 3: Local answer
        Generate a response for a specific community based on the query and community summary.

        Args:
            community (Dict[str, Any]): Community information with a summary.
            query (str): The local query to be answered based on community-level information.

        Returns:
            str: The final synthesized local answer.
        """
        try:
            local_response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Generate a response for the following community based on the query and community summary.
                        The response must:
                        - Be concise yet informative.
                        - Address the query directly.
                        - Utilize the community summary to provide context.
                        - Ensure logical coherence and relevance.
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\nCommunity Summary: {community['summary']}",
                    },
                ],
            )

            return local_response.choices[0].message.content

        except Exception as e:
            print(f"Error generating local answer: {e}")
            return "Unable to generate a local answer at this time."

    def global_answer(
        self, communities_summarization: List[Dict[str, Any]], query: str
    ) -> str:
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
            local_responses = []
            for community in communities_summarization:
                local_response = self.local_answer(community, query)
                local_responses.append(local_response)

            global_response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Generate a response for the following communities based on the query and community summaries.
                        The response must:
                        - Be concise yet informative.
                        - Address the query directly.
                        - Utilize the community summaries to provide context.
                        - Ensure logical coherence and relevance.
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\nCommunities: {communities_summarization}",
                    },
                ],
            )

            return global_response.choices[0].message.content
        except Exception as e:
            print(f"Error generating global answer: {e}")
            return "Unable to generate a global answer at this time."

    async def execute_pipeline(self, query: str):
        """
        Full pipeline execution starting from recommendations to insights.

        Args:
            query (str): Natural language query for the pipeline.
        """
        print("Executing pipeline...")
        recommendations = self.hybrid_recommendations(query)

        return recommendations


if __name__ == "__main__":
    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    graph_rag = RecommendationPipeline(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
    )

    query = "Find hotels with TV and Parking for a long family stay."

    asyncio.run(graph_rag.execute_pipeline(query))

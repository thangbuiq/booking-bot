import logging
from typing import Any, Dict, List

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

from core.agent import FuncationCallingAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RecommendationPipeline")


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

        Args:
            openai_model (str): OpenAI model name. Defaults to "gpt-4o-mini".
            neo4j_uri (str): Neo4j URI.
            neo4j_username (str): Neo4j username.
            neo4j_password (str): Neo4j password.
        """
        self.openai_model = openai_model
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to parse a user query and extract parameters based on predefined lists.

        Args:
            query (str): Natural language query.

        Returns:
            Dict[str, Any]: Extracted parameters.
        """
        logger.info("Starting recommendation pipeline.")
        try:
            # Initialize the agent with the OpenAI LLM and custom tools
            self.agent = FuncationCallingAgent(
                llm=OpenAI(self.openai_model),
                tools=[
                    FunctionTool.from_defaults(self.process_cypher_recommendations),
                    FunctionTool.from_defaults(self.process_graph_rag_recommendations),
                    FunctionTool.from_defaults(self.hybrid_recommendations),
                ],
                timeout=120,
                verbose=True,
            )

            # Get the agent response
            agent_response = await self.agent.run(input=query)

            logger.info("Pipeline completed successfully.")
            return agent_response["tool_output"]
        except Exception as e:
            logger.error(f"Error running recommendation pipeline: {e}")
            raise

    def process_cypher_recommendations(
        self, amenities: List[str], stay_type: str, stay_duration: str
    ) -> List[Dict[str, Any]]:
        """
        Process Cypher-based recommendations.

        Params:
            - amenities: A list of required amenities.
            - stay_type: Type of stay (e.g., Couple, Family).
            - stay_duration: Duration of stay (e.g., Short, Medium).

        Returns:
            List[Dict[str, Any]]: Structured element instances.
        """
        logger.info("Processing Cypher recommendations.")
        from core.recommendations.cypher_graph import CypherGraphHotelRecommender

        try:
            recommender = CypherGraphHotelRecommender(
                self.neo4j_uri, self.neo4j_username, self.neo4j_password
            )

            recommended_hotels = recommender.recommend_hotels(
                amenities=amenities, stay_type=stay_type, stay_duration=stay_duration
            )
            recommender.close()
            return recommended_hotels
        except Exception as e:
            logger.error(f"Error in Cypher recommendations: {e}")
            return []

    def process_graph_rag_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """
        Process recommendations using the Graph-based RAG method.

        Params:
            - query: Natural language query.

        Returns:
            List[Dict[str, Any]]: Structured element instances.
        """
        logger.info("Processing Graph RAG recommendations.")
        from core.recommendations.graph_rag import GraphRAGHotelRecommender

        try:
            recommender = GraphRAGHotelRecommender(
                self.neo4j_uri, self.neo4j_username, self.neo4j_password
            )

            recommended_hotels = recommender.recommend_hotels(query=query)
            recommender.close()
            return recommended_hotels
        except Exception as e:
            logger.error(f"Error in Graph RAG recommendations: {e}")
            return []

    def hybrid_recommendations(
        self,
        cypher_recommendations: List[Dict[str, Any]],
        graph_rag_recommendations: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Combine and rerank the recommendations from Cypher and Graph RAG methods.

        Params:
            - cypher_recommendations: Recommendations from the Cypher method.
            - graph_rag_recommendations: Recommendations from the Graph RAG method.
            - query: The original query for context-aware reranking.

        Returns:
            List[Dict[str, Any]]: Reranked recommendations.
        """
        logger.info("Combining recommendations using hybrid approach.")
        try:
            # Combine the recommendations
            combined = cypher_recommendations + graph_rag_recommendations

            # Prepare data for LLM-based reranking
            prompt = (
                "You are an AI assistant specializing in hotel recommendations. "
                "Given the user query and the list of recommendations, rank the recommendations "
                "from most to least relevant based on the user's preferences.\n\n"
                f"User Query: {query}\n\n"
                "Recommendations:\n"
            )

            for i, rec in enumerate(combined, 1):
                prompt += f"{i}. {rec}\n"

            prompt += "\nReturn the ranked recommendations as a numbered list."

            # Use the LLM to rerank
            llm = OpenAI(self.openai_model)
            response = llm.complete(prompt)

            # Parse the response into a ranked list
            ranked_indices = [
                int(line.split(".")[0]) - 1
                for line in response.split("\n")
                if line.strip() and line.split(".")[0].isdigit()
            ]

            reranked = [combined[i] for i in ranked_indices if 0 <= i < len(combined)]
            return reranked
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return []

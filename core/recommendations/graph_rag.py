from typing import Any, Dict, List
import openai
from core.recommendations.base import BaseHotelRecommender
from neo4j.exceptions import Neo4jError


class GraphRAGHotelRecommender(BaseHotelRecommender):
    def __init__(self, uri: str, username: str, password: str, openai_model: str):
        """
        Initialize the GraphRAGHotelRecommender.

        Args:
            uri (str): URI for the Neo4j database.
            username (str): Username for the Neo4j database.
            password (str): Password for the Neo4j database.
            openai_model (str): OpenAI model to use for LLM operations.
        """
        super().__init__(uri=uri, username=username, password=password)
        self.openai_model = openai_model

    def communities_detection(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Detect top n communities within the graph database using Neo4j's community detection algorithms.

        Args:
            n (int): The number of communities to detect.

        Returns:
            List[Dict[str, Any]]: List of detected communities with their respective nodes.
        """
        try:
            with self.driver.session() as session:
                query = f"""
                CALL gds.louvain.stream({{
                    graphName: 'yourGraphName',
                    maxCommunities: {n}
                }})
                YIELD nodeId, communityId AS community
                RETURN gds.util.asNode(nodeId).name AS node, community
                ORDER BY community
                """
                result = session.run(query)
                communities = {}
                for record in result:
                    community = record["community"]
                    node = record["node"]
                    communities.setdefault(community, []).append(node)

            return [{"community": k, "nodes": v} for k, v in communities.items()]
        except Neo4jError as e:
            raise RuntimeError(f"Error during community detection: {e}")

    def communities_summarization(self, communities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Summarize detected communities using LLM.

        Args:
            communities (List[Dict[str, Any]]): List of detected communities.

        Returns:
            List[Dict[str, Any]]: List of summaries for each community.
        """
        summaries = []
        for community in communities:
            try:
                nodes = ", ".join(community["nodes"])
                prompt = (
                    "You are an AI assistant specializing in hotel recommendations. "
                    "A community represents a group of hotels or places with shared characteristics. "
                    "Summarize the characteristics of the hotels or places in this community based on the provided list of nodes. "
                    "Be concise, relevant, and focus on recommendation purposes.\n\n"
                    f"Community Nodes (hotels or places): {nodes}\n\n"
                    "Provide a summary of the community for recommendation:"
                )
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": prompt}
                    ],
                )
                summaries.append({
                    "community": community["community"],
                    "summary": response.choices[0].message.content.strip(),
                })
            except openai.error.OpenAIError as e:
                raise RuntimeError(f"Error summarizing community {community['community']}: {e}")

        return summaries

    def local_answer(self, community: Dict[str, Any], query: str) -> str:
        """
        Generate a recommendation-specific response for a specific community.

        Args:
            community (Dict[str, Any]): Community summary.
            query (str): User query.

        Returns:
            str: Generated recommendation response.
        """
        try:
            prompt = (
                "You are an AI assistant specializing in personalized hotel recommendations. "
                "Using the provided community summary and the user's preferences from their query, "
                "recommend suitable hotels or places that best match the user's needs.\n\n"
                f"User Query: {query}\nCommunity Summary: {community['summary']}\n\n"
                "Generate a recommendation response:"
            )
            local_response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": prompt}
                ],
            )
            return local_response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            raise RuntimeError(f"Error generating local answer for community {community['community']}: {e}")

    def global_answer(self, communities_summarization: List[Dict[str, Any]], query: str) -> str:
        """
        Synthesize a global recommendation response from community-level summaries.

        Args:
            communities_summarization (List[Dict[str, Any]]): Summarized communities.
            query (str): User query.

        Returns:
            str: Synthesized global recommendation response.
        """
        try:
            summaries = "\n".join(
                f"Community {community['community']}: {community['summary']}"
                for community in communities_summarization
            )
            prompt = (
                "You are an AI assistant tasked with synthesizing a comprehensive hotel recommendation response. "
                "Use the provided community summaries and the user's preferences to create a coherent global answer. "
                "Address the user's query by recommending hotels or places across all communities.\n\n"
                f"User Query: {query}\n\nCommunity Summaries:\n{summaries}\n\n"
                "Generate the global recommendation response:"
            )
            global_response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": prompt}
                ],
            )
            return global_response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            raise RuntimeError(f"Error generating global answer: {e}")

    def recommend_hotels(self, query: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Recommend hotels to users using the GraphRAG method.

        Args:
            query (str): User query.
            n (int): The number of communities to detect.

        Returns:
            List[Dict[str, Any]]: List of recommended hotels.
        """
        try:
            communities = self.communities_detection(n=n)
            communities_summaries = self.communities_summarization(communities)
            recommendations = self.global_answer(communities_summaries, query)
            return recommendations
        except Exception as e:
            raise RuntimeError(f"Error in recommending hotels: {e}")

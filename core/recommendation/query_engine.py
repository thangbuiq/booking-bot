import re
from typing import Any
from typing import Dict
from typing import List

from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.query_engine import CustomQueryEngine
from recommendation.constants import ENTITIES_GRAPH_REGEXP_PATTERN
from recommendation.constants import TO_BE_CLEANED_RESPONSE
from recommendation.storage import RecommendationGraphStore


class RecommendationGraphRAGQueryEngine(CustomQueryEngine):
    graph_store: RecommendationGraphStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        """
        Custom query to retrieve recommendations using knowledge from the graph.

        Args:
            query_str (str): Query string.

        Returns:
            str: Recommendations.
        """
        entities = self.get_entities(
            query_str=query_str, similarity_top_k=self.similarity_top_k
        )

        community_ids = self.retrieve_entity_communities(
            entity_info=self.graph_store.entity_info, entities=entities
        )

        community_recommendations = []
        for community_id in community_ids:
            recommendations = self._generate_community_recommendations(
                community_id=community_id, query_str=query_str, entities=entities
            )
            community_recommendations.append(recommendations)

        # Aggregate and rank recommendations
        final_recommendations = self._aggregate_recommendations(
            community_recommendations=community_recommendations, query_str=query_str
        )

        return self._format_recommendations(recommendations=final_recommendations)

    def get_entities(self, query_str: str, similarity_top_k: str) -> List[str]:
        """
        Get entities from the graph.

        Args:
            query_str (str): Query string.
            similarity_top_k (str): Similarity top k.

        Returns:
            List[str]: List of entities.
        """
        retrieved_nodes = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        entities = set()
        for node in retrieved_nodes:
            matches = re.findall(
                pattern=ENTITIES_GRAPH_REGEXP_PATTERN,
                string=node,
                flags=re.MULTILINE | re.IGNORECASE,
            )

            for match in matches:
                subject = match[0]
                obj = match[2]
                entities.add(subject)
                entities.add(obj)

        return list(entities)

    def retrieve_entity_communities(
        self, entity_info: Dict[str, Any], entities: List[str]
    ) -> List[str]:
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
            entity_info (Dict[str, Any]): Entity information.
            entities (List[str]): List of entities.

        Returns:
            List[str]: List of cluster information.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def _generate_community_recommendations(
        self, community_id: int, query_str: str, entities: List[str]
    ) -> str:
        """
        Generate recommendations from a specific community.

        Args:
            community_id (int): The community ID.
            query_str (str): The query string.
            entities (List[str]): The entities.

        Returns:
            str: The recommendations.
        """
        community_summary = self.graph_store.community_summaries.get(community_id, "")
        entities_str = ", ".join(entities)
        messages = [
            ChatMessage(
                role="system",
                content=(
                    f"Given the community information below and the query, generate relevant "
                    f"recommendations. Focus on items that match the query intent and have "
                    f"strong relationships within the community.\n\n"
                    f"Community Summary: {community_summary}\n"
                    f"Query Entities: {entities_str}"
                ),
            ),
            ChatMessage(role="user", content=query_str),
        ]

        final_response = self.llm.chat(messages=messages)
        cleaned_final_response = re.sub(
            TO_BE_CLEANED_RESPONSE, "", str(final_response)
        ).strip()
        return cleaned_final_response

    def _aggregate_recommendations(
        self, community_recommendations: List[str], query_str: str
    ) -> str:
        """
        Aggregate and rank recommendations from different communities.

        Args:
            community_recommendations (List[str]): The community recommendations.
            query_str (str): The query string.

        Returns:
            str: The aggregated recommendations
        """
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Combine and prioritize the following recommendations based on "
                    "relevance to the query, relationship strength, and diversity. "
                    "Provide a clear explanation for each recommendation."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Query: {query_str}\n\n"
                    f"Community Recommendations:\n"
                    f"{'-' * 40}\n" + "\n".join(community_recommendations)
                ),
            ),
        ]

        final_response = self.llm.chat(messages=messages)
        cleaned_final_response = re.sub(
            TO_BE_CLEANED_RESPONSE, "", str(final_response)
        ).strip()
        return cleaned_final_response

    def _format_recommendations(self, recommendations: str) -> str:
        """
        Format recommendations for presentation.

        Args:
            recommendations (str): The recommendations.

        Returns:
            str: The formatted recommendations.
        """
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Format the recommendations in a clear, structured way. "
                    "Include relevant details such as similarity scores, key features, "
                    "and reasoning for each recommendation."
                ),
            ),
            ChatMessage(role="user", content=recommendations),
        ]

        final_response = self.llm.chat(messages=messages)
        cleaned_final_response = re.sub(
            TO_BE_CLEANED_RESPONSE, "", str(final_response)
        ).strip()
        return cleaned_final_response

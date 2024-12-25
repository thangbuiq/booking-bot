import re
from typing import Any
from typing import Dict
from typing import List

from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.query_engine import CustomQueryEngine

from recommendation.constants import ENTITIES_REGEXP_PATTERN
from recommendation.storage import GraphRAGStore


class RecommendationGraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
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
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(
                community_summary=community_summary, query_str=query_str
            )
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]

        final_answers = self.aggregate_answers(
            query_str=query_str, community_answers=community_answers
        )
        return final_answers

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
                pattern=ENTITIES_REGEXP_PATTERN,
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

    def generate_answer_from_summary(
        self, community_summary: Dict[str, Any], query_str: str
    ) -> str:
        """
        Generate answer from community summary.

        Args:
            community_summary (Dict[str, Any]): Community summary.
            query_str (str): Query string.

        Returns:
            str: Answer.
        """
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query_str}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        final_response = self.llm.chat(messages=messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response

    def aggregate_answers(self, query_str: str, community_answers: List[str]) -> str:
        """
        Aggregate answers.

        Args:
            query_str (str): Query string.
            community_answers (List[str]): List of community answers.

        Returns:
            str: Aggregated answer.
        """
        prompt = f"Given the query: {query_str} and the intermediate answers, combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response

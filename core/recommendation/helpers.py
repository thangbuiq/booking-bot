import re
from typing import Any
from typing import List
from typing import Tuple

import pandas as pd
from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import LLM
from llama_index.core.schema import TextNode

from core.recommendation.constants import ENTITIES_RESPONSE_PATTERN
from core.recommendation.constants import RECOMMENDATION_KG_EXTRACT_TMPL
from core.recommendation.constants import RELATIONSHIPS_RESPONSE_PATTERN
from core.recommendation.extractor import RecommendationGraphExtractor
from core.recommendation.query_engine import RecommendationGraphRAGQueryEngine
from core.recommendation.secrets import NEO4J_PASSWORD
from core.recommendation.secrets import NEO4J_URL
from core.recommendation.secrets import NEO4J_USERNAME
from core.recommendation.storage import RecommendationGraphStore


def parse_fn(response_str: str) -> Tuple[List[Any], List[Any]]:
    """
    Parse the response from the language model.

    Args:
        response_str (str): The response from the language model.

    Returns:
        Tuple[List[Any], List[Any]]: A tuple of entities and relationships.
    """
    entities = [
        (name, type_, desc, attrs)
        for name, type_, desc, attrs in re.findall(
            ENTITIES_RESPONSE_PATTERN, response_str
        )
    ]
    relationships = [
        (src, tgt, rel, strength, desc, features)
        for src, tgt, rel, strength, desc, features in re.findall(
            RELATIONSHIPS_RESPONSE_PATTERN, response_str
        )
    ]

    return entities, relationships


def read_data(file_path: str) -> List[TextNode]:
    """
    Read the data from a file.

    Args:
        file_path (str): The file path.

    Returns:
        List[TextNode]: The list of text nodes.
    """
    bookings = pd.read_parquet(file_path)
    nodes = [
        TextNode(text=f"{row['title']}: {row['text']}")
        for i, row in bookings.iterrows()
    ]

    return nodes


def create_recommendation_system(
    nodes: List[TextNode], llm: LLM
) -> RecommendationGraphRAGQueryEngine:
    """
    Create and initialize the recommendation system.

    Args:
        nodes (List[TextNode]): The list of text nodes.
        llm (LLM): The language model.

    Returns:
        RecommendationGraphRAGQueryEngine: The recommendation system.
    """

    # Initialize graph components
    graph_store = RecommendationGraphStore(
        username=NEO4J_USERNAME, password=NEO4J_PASSWORD, url=NEO4J_URL
    )

    # Create knowledge graph extractor
    kg_extractor = RecommendationGraphExtractor(
        llm=llm,
        extract_prompt=RECOMMENDATION_KG_EXTRACT_TMPL,
        parse_fn=parse_fn,
        max_paths_per_chunk=5,
    )

    # Build property graph index
    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )

    # Create query engine
    query_engine = RecommendationGraphRAGQueryEngine(
        graph_store=graph_store, index=index, llm=llm, similarity_top_k=10
    )

    return query_engine

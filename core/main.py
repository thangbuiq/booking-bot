import pandas as pd
from llama_index.core import PropertyGraphIndex
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI

from core.recommendation.constants import KG_TRIPLET_EXTRACT_TMPL
from core.recommendation.extractor import GraphRAGExtractor
from core.recommendation.helpers import parse_fn
from core.recommendation.query_engine import RecommendationGraphRAGQueryEngine
from core.recommendation.secrets import NEO4J_PASSWORD
from core.recommendation.secrets import NEO4J_URL
from core.recommendation.secrets import NEO4J_USERNAME
from core.recommendation.storage import GraphRAGStore

llm = OpenAI(model="gpt-4o-mini")

kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    parse_fn=parse_fn,
    max_paths_per_chunk=2,
)

graph_store = GraphRAGStore(
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    url=NEO4J_URL,
)

bookings = pd.read_parquet("data/bookings.parquet")
nodes = [
    TextNode(text=f"{row['title']}: {row['text']}") for i, row in bookings.iterrows()
]

index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)

query_engine = RecommendationGraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    llm=llm,
    index=index,
    similarity_top_k=10,
)

response = query_engine.query("What are the main news discussed in the document?")
print(response)

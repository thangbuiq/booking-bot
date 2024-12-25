import asyncio
from collections.abc import Callable
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import nest_asyncio
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.graph_stores.types import KG_NODES_KEY
from llama_index.core.graph_stores.types import KG_RELATIONS_KEY
from llama_index.core.graph_stores.types import Relation
from llama_index.core.indices.property_graph import default_parse_triplets_fn
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import BaseNode
from llama_index.core.schema import TransformComponent

nest_asyncio.apply()


class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Optional[Callable] = default_parse_triplets_fn,
        max_paths_per_chunk: int = 100,
        num_workers: int = 4,
    ):
        """
        Initialize the GraphRAGExtractor.

        Args:
            llm (LLM): The language model to use.
            extract_prompt (Union[str, PromptTemplate]): The prompt to use for extracting triples.
            parse_fn (callable): A function to parse the output of the language model.
            num_workers (int): The number of workers to use for parallel processing.
            max_paths_per_chunk (int): The maximum number of paths to extract per chunk.
        """
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(template=extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Extract triples from a list of nodes.

        Args:
            nodes (List[BaseNode]): The nodes to extract triples from.
            show_progress (bool): Whether to show the progress of the extraction.

        Returns:
            List[BaseNode]: The nodes with extracted triples.
        """
        return asyncio.run(
            self.acall(nodes=nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """
        Extract triples from a node asynchronously.

        Args:
            node (BaseNode): The node to extract triples from.

        Returns:
            BaseNode: The node with extracted triples.
        """
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_reponse = await self.llm.apredict(
                prompt=self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_reponse)
        except ValueError:
            entities, entities_relationship = [], []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity,
                label=entity_type,
                properties=entity_metadata,
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Extract triples from a list of nodes asynchronously.

        Args:
            nodes (List[BaseNode]): The nodes to extract triples from.
            show_progress (bool): Whether to show the progress of the extraction.

        Returns:
            List[BaseNode]: The nodes with extracted triples.
        """
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node=node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

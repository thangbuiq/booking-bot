import re
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Tuple

import networkx as nx
from graspologic.partition import hierarchical_leiden
from graspologic.partition import HierarchicalClusters
from llama_index.core.llms import ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI


class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
    max_cluster_size = 5

    def generate_community_summary(self, text: str) -> str:
        """
        Generate a summary for the community.

        Args:
            text (str): Text to summarize.

        Returns:
            str: Summary of the text.
        """
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = OpenAI(model="gpt-4o-mini").chat(messages=messages)
        clean_reponse = re.sub(r"^assistant:\s*", "", str(response)).strip()

        return clean_reponse

    def build_communities(self) -> None:
        """
        Build communities from the graph and summarize them.
        """
        graph = self._build_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            graph=graph, clusters=community_hierarchical_clusters
        )
        self._summarize_communities(community_info=community_info)

    def _build_nx_graph(self) -> nx.Graph:
        """
        Build a NetworkX graph from the graph store.

        Returns:
            nx.Graph: A NetworkX graph.
        """
        graph = nx.Graph()
        return graph

    def _collect_community_info(
        self, graph: nx.Graph, clusters: HierarchicalClusters
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Collects information about the communities.

        Args:
            graph (nx.Graph): A NetworkX graph.
            clusters (HierarchicalClusters): Hierarchical clusters.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing entity information and community information.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            # Update entity info
            entity_info[node].add(cluster_id)

            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        # Convert sets to lists for easier serialization if needed
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info: Dict[str, Any]) -> None:
        """
        Generate summaries for the communities.

        Args:
            community_info (Dict[str, Any]): Community information.
        """
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = self.generate_community_summary(
                text=details_text
            )

    def get_community_summaries(self):
        """
        Get community summaries.

        Returns:
            Dict[str, Any]: Community summaries.
        """
        if not self.community_summary:
            self.build_communities()

        return self.community_summary


# class GraphBuilder:
#     def __init__(
#         self,
#         uri: str,
#         restaurants: Optional[List[Dict[str, Any]]] = None,
#         username: str = "",
#         password: str = "",
#         build_network: bool = False,
#     ):
#         """
#         Constructor for GraphBuilder class.

#         Args:
#             uri (str): URI of the Neo4j database.
#             restaurants (Optional[List[Dict[str, Any]]]): A list of dictionaries containing the restaurant details.
#             username (str): Username of the Neo4j database. Default is an empty string.
#             password (str): Password of the Neo4j database. Default is an empty string.
#             build_network (bool): A flag to build the graph network. Default is False.
#         """
#         try:
#             self._driver = GraphDatabase.driver(
#                 uri=uri,
#                 auth=(username, password),
#             )
#             self._restaurants = restaurants
#             self._build_network = build_network
#         except Exception as e:
#             logger.error(f"Failed to connect to Neo4j database: {e}")
#             raise e

#     def _create_constraints(self):
#         """
#         Creates constraints for unique nodes.
#         """
#         with self._driver.session() as session:
#             try:
#                 session.run(
#                     "CREATE CONSTRAINT restaurant_url IF NOT EXISTS FOR (r:Restaurant) REQUIRE r.url IS UNIQUE"
#                 )
#                 session.run(
#                     "CREATE CONSTRAINT cuisine_name IF NOT EXISTS FOR (c:Cuisine) REQUIRE c.name IS UNIQUE"
#                 )
#                 session.run(
#                     "CREATE CONSTRAINT reviewer_type IF NOT EXISTS FOR (rt:ReviewerType) REQUIRE rt.type IS UNIQUE"
#                 )

#                 logger.info("Constraints created successfully!")
#             except Exception as e:
#                 logger.error(f"Failed to create constraints: {e}")
#                 raise e

#     def _create_restaurant(self, restaurant: Dict[str, Any]) -> None:
#         """
#         Creates a restaurant node in the graph.

#         Args:
#             restaurant (Dict[str, Any]): A dictionary containing the restaurant details.
#         """
#         try:
#             with self._driver.session() as session:
#                 # Extract name from URL or use the name provided
#                 restaurant_name = restaurant.get(
#                     "name",
#                     restaurant.get("url", "").split("-")[-1]
#                     if "-" in restaurant.get("url", "")
#                     else "Unknown",
#                 )

#                 # Create a restaurant node
#                 query = """
#                 MERGE (r:Restaurant {url: $url})
#                 SET r.name = $name,
#                     r.address = $address,
#                     r.latitude = $lat,
#                     r.longitude = $long,
#                     r.price_range = $price_range,
#                     r.ranking = $ranking,
#                     r.rating = $rating,
#                     r.review_count = $review_count
#                 RETURN r
#                 """

#                 session.run(
#                     query=query,
#                     url=restaurant.get("url", ""),
#                     name=restaurant_name,
#                     address=restaurant.get("address", ""),
#                     lat=restaurant.get("latitude", None),
#                     long=restaurant.get("longitude", None),
#                     price_range=restaurant.get("price_range", ""),
#                     ranking=restaurant.get("ranking", None),
#                     rating=restaurant.get("rating", None),
#                     review_count=restaurant.get("review_count", 0),
#                 )

#                 # Create cuisine nodes and relationships
#                 for cuisine in restaurant.get("cuisines", []):
#                     query = """
#                     MERGE (c:Cuisine {type: $cuisine})
#                     WITH c
#                     MATCH (r:Restaurant {url: $url})
#                     MERGE (r)-[:HAS_CUISINE]->(c)
#                     """

#                     session.run(
#                         query=query,
#                         cuisine=cuisine,
#                         url=restaurant.get("url", ""),
#                     )

#                 # Create reviews
#                 for review in restaurant.get("reviews", []):
#                     query = """
#                     MATCH (r:Restaurant {url: $url})
#                     MERGE (rt:ReviewerType {type: $review_type})
#                     CREATE (rev:Review {
#                         title: $title,
#                         text: $text,
#                         rating: $rating,
#                         date: $review_date,
#                         review_type: $review_type
#                     })
#                     CREATE (rev)-[:ABOUT]->(r)
#                     CREATE (rev)-[:BY]->(rt)
#                     """

#                     session.run(
#                         query=query,
#                         url=restaurant.get("url", ""),
#                         review_type=review.get("review_type", "Unknown"),
#                         title=review.get("title", ""),
#                         text=review.get("text", ""),
#                         rating=review.get("rating", None),
#                         review_date=review.get(
#                             "review_date", pendulum.now().isoformat()
#                         ),
#                     )

#                 logger.info(f"Restaurant {restaurant_name} processed successfully!")

#         except Exception as e:
#             logger.error(f"Failed to create restaurant node: {e}")
#             raise e

#     def _create_restaurants(self) -> None:
#         """
#         Creates restaurant nodes in the graph.

#         Args:
#             restaurants (Dict[str, Any]): A dictionary containing the restaurant details.
#         """
#         start_time = pendulum.now()
#         successfull_creation = 0
#         failed_creation = 0

#         for restaurant in self._restaurants:
#             try:
#                 self._create_restaurant(restaurant)
#                 successfull_creation += 1
#             except Exception as e:
#                 logger.error(f"Failed to create restaurant node: {e}")
#                 failed_creation += 1

#         end_time = pendulum.now()
#         logger.info(
#             f"Processed {successfull_creation} restaurants in {end_time.diff(start_time).in_seconds()} seconds."
#         )
#         logger.info(f"Failed to process {failed_creation} restaurants.")

#     def _build_graph(self) -> nx.Graph:
#         """
#         Builds the graph network by querying the Neo4j database.

#         Returns:
#             nx.Graph: A NetworkX graph object.
#         """
#         try:
#             with self._driver.session() as session:
#                 query = """
#                 MATCH (r:Restaurant)-[:HAS_CUISINE]->(c:Cuisine)
#                 RETURN r.url AS restaurant, c.type AS cuisine
#                 """

#                 result = session.run(query=query)

#                 g = nx.Graph()

#                 for record in result:
#                     g.add_node(record["restaurant"], label="Restaurant")
#                     g.add_node(record["cuisine"], label="Cuisine")
#                     g.add_edge(record["restaurant"], record["cuisine"])

#                 logger.info("Graph network built successfully!")

#                 return g
#         except Exception as e:
#             logger.error(f"Failed to build graph network: {e}")
#             raise

#     def build(self) -> Optional[nx.Graph]:
#         """
#         Builds the graph network by creating constraints and nodes.
#         """
#         self._create_constraints()
#         self._create_restaurants()

#         if self._build_network:
#             return self._build_graph()

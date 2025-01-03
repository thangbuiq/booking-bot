from typing import Any, Dict, List

import openai
from neo4j.exceptions import Neo4jError

from core.recommendations.base import BaseHotelRecommender


class GraphRAGHotelRecommender(BaseHotelRecommender):
    def __init__(
        self, uri: str, username: str, password: str, openai_model: str = "gpt-4o-mini"
    ):
        """
        Initialize the GraphRAGHotelRecommender.

        Args:
            uri (str): URI for the Neo4j database.
            username (str): Username for the Neo4j database.
            password (str): Password for the Neo4j database.
            openai_model (str): OpenAI model to use for LLM operations. Defaults to "gpt-4o-mini".
        """
        super().__init__(uri=uri, username=username, password=password)
        self.openai_model = openai_model

        # Detect communities and summarize them
        self.communities = self.communities_detection()
        self.summaries = self.communities_summarization(self.communities)

    def _create_constraints(self):
        """
        Create constraints in the Neo4j database.
        """
        return super()._create_constraints()

    def communities_detection(
        self, graph_name: str = "hotelCommunityGraph"
    ) -> Dict[str, Any]:
        """
        Detect communities within the graph database using Louvain algorithm.

        Args:
            graph_name (str): Name of the graph projection in Neo4j.

        Returns:
            Dict[str, Any]: Communities with their nodes and relationships.
        """
        try:
            with self.driver.session() as session:
                # Check and create graph if necessary
                result = session.run(f"""
                    CALL gds.graph.exists('{graph_name}')
                    YIELD exists
                    RETURN exists
                """)
                graph_exists = result.single()["exists"]

                if not graph_exists:
                    session.run(f"""
                        CALL gds.graph.project(
                            '{graph_name}',
                            '*',
                            '*'
                        )
                    """)

                # Run Louvain community detection
                session.run(f"""
                    CALL gds.louvain.write('{graph_name}', {{
                        writeProperty: 'communityId'
                    }})
                """)

                # Fetch communities with relationships
                result = session.run("""
                    MATCH (n)
                    WHERE n.communityId IS NOT NULL
                    WITH DISTINCT n.communityId AS community
                    MATCH (n1)-[r]-(n2)
                    WHERE n1.communityId = community AND n2.communityId = community
                    WITH community,
                         COLLECT(DISTINCT {
                             id: id(n1),
                             labels: labels(n1),
                             properties: properties(n1)
                         }) + COLLECT(DISTINCT {
                             id: id(n2),
                             labels: labels(n2),
                             properties: properties(n2)
                         }) AS nodes,
                         COLLECT(DISTINCT {
                             type: type(r),
                             properties: properties(r),
                             source: id(startNode(r)),
                             target: id(endNode(r))
                         }) AS relationships
                    RETURN community, nodes, relationships
                    ORDER BY community
                """)

                communities = {}
                for record in result:
                    unique_nodes = {
                        node["id"]: node for node in record["nodes"]
                    }.values()
                    communities[record["community"]] = {
                        "nodes": list(unique_nodes),
                        "relationships": record["relationships"],
                    }
                return communities

        except Neo4jError as e:
            raise RuntimeError(f"Error during community detection: {e}")

    def communities_summarization(
        self, communities: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Summarize detected communities using LLM.

        Args:
            communities (Dict[str, Any]): Communities data.

        Returns:
            List[Dict[str, Any]]: Summaries for each community.
        """
        summaries = []
        for community_id, community_data in communities.items():
            try:
                nodes = ", ".join(
                    f"{node['properties'].get('name', 'Unknown')} ({', '.join(node['labels'])})"
                    for node in community_data["nodes"]
                )
                prompt = (
                    "You are an AI assistant specializing in hotel recommendations. "
                    "Summarize the characteristics of the following hotels or places "
                    "to support recommendations:\n\n"
                    f"Community Nodes: {nodes}\n\n"
                    "Provide a summary:"
                )
                response = openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "system", "content": prompt}],
                )
                summaries.append(
                    {
                        "community": community_id,
                        "summary": response.choices[0].message.content.strip(),
                    }
                )
            except Exception as e:
                raise RuntimeError(f"Error summarizing community {community_id}: {e}")
        return summaries

    def recommend_hotels(self, query: str) -> str:
        """
        Generate hotel recommendations based on the user's query.

        Args:
            query (str): User query.

        Returns:
            str: Recommendation response.
        """
        try:
            summary_text = "\n".join(
                f"Community {s['community']}: {s['summary']}" for s in self.summaries
            )
            prompt = (
                "Using the following community summaries, generate a hotel recommendation "
                "based on the user's query:\n\n"
                f"User Query: {query}\n\nCommunity Summaries:\n{summary_text}\n\n"
                "Provide recommendations:"
            )
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "system", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error in recommending hotels: {e}")


if __name__ == "__main__":
    recommender = GraphRAGHotelRecommender(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        openai_model="gpt-4o-mini",
    )
    query = "I am looking for a hotel with a air conditioning and TV."
    recommendation = recommender.recommend_hotels(query)
    print(recommendation)

from typing import Any, Dict

import community as community_louvain
import networkx as nx
import numpy as np
from pyvis import Network
from recommendation.constants import GRAPH_NETWORK_HTML_FILEPATH


class GraphVisualization:
    def __init__(
        self,
        g: nx.Graph,
        partition: Dict[str, Any] = None,
        title: str = "Visualization of Graph Network",
        file_path: str = GRAPH_NETWORK_HTML_FILEPATH,
    ):
        """
        Constructor for GraphVisualization class.

        Args:
            g (nx.Graph): A NetworkX graph object.
            partition (Dict[str, Any]): A dictionary containing the community partition of the graph.
            title (str): Title of the graph network visualization.
            file_path (str): File path to save the graph network visualization.
        """
        if not isinstance(g, nx.Graph):
            raise ValueError("The input graph must be a NetworkX graph object.")

        self._g = g
        self._partition = partition
        self._title = title
        self._file_path = file_path

    @property
    def partition(self) -> Dict[str, Any]:
        """
        Getter for the partition property.
        """
        return self._partition

    @partition.setter
    def partition(self, partition: Dict[str, Any]) -> None:
        self._partition = partition

    def visualize(self) -> str:
        """
        Visualizes the graph network using the PyVis library.

        Returns:
            str: File path to the saved HTML file.
        """
        net = Network(height="800px", width="100%", notebook=True, heading=self.title)
        net.show_buttons()

        # Assign colors based on communities
        if self._partition:
            unique_communities = set(self._partition.values())
            color_map = {
                community: f"#{hash(community) % 0xFFFFFF:06x}"
                for community in unique_communities
            }
        else:
            color_map = None

        # Construct the nodes
        for node in self._g.nodes:
            # Node size based on degree
            degree = self._g.degree[node]
            net.add_node(
                node,
                label=str(node),
                color=color_map[self._partition[node]] if color_map else "#000000",
                size=degree * 2,
            )

        # Construct the edges
        for edge in self._g.edges:
            net.add_edge(edge[0], edge[1])

        net.show(self._file_path)

        return self._file_path


class GraphAnalyzer:
    def __init__(self, g: nx.Graph):
        """
        Constructor for GraphAnalyzer class.

        Args:
            g (nx.Graph): A NetworkX graph object.
        """
        if not isinstance(g, nx.Graph):
            raise ValueError("The input graph must be a NetworkX graph object.")

        self._g = g
        self._gv = GraphVisualization(g)

    def _analyze_communities(self) -> Dict[str, Any]:
        """
        Analyze detected communities and provide additional insights.

        Returns:
            Dict[str, Any]: A dictionary containing the community insights.
        """
        communities = {}
        partition = self._gv.partition
        for community_id in set(partition.values()):
            # Nodes in this community
            community_nodes = [
                node for node, comm in partition.items() if comm == community_id
            ]

            # Subgraph for this community
            community_subgraph = self._g.subgraph(community_nodes)

            communities[community_id] = {
                "num_nodes": len(community_nodes),
                "num_edges": community_subgraph.number_of_edges(),
                "avg_degree": np.mean(
                    [deg for node, deg in community_subgraph.degree()]
                ),
                "density": nx.density(community_subgraph),
            }

        return communities

    def detect_communities(self, method: str = "louvain") -> Dict[str, Any]:
        """
        Detects communities in the graph network using the Louvain algorithm.

        Args:
            method (str): The method used to detect communities. Default is 'louvain'.

        Returns:
            Dict[str, Any]: A dictionary containing the community partition of the graph.
        """
        # Community detection methods
        if method == "louvain":
            partition = community_louvain.best_partition(self._g)
        elif method == "girvan_newman":
            comp = list(nx.community.girvan_newman(self._g))
            partition = {
                node: idx
                for idx, community_set in enumerate(comp[0])
                for node in community_set
            }
        elif method == "label_propagation":
            partition = nx.community.label_propagation_communities(self._g)
            partition = {
                node: idx for idx, comm in enumerate(partition) for node in comm
            }
        else:
            raise ValueError(f"Invalid community detection method: {method}")

        # Update partition property for visualization
        self._gv.partition = partition

        # Plot the graph network
        self._gv.visualize()

        # Additional community analysis
        community_insights = self._analyze_communities()

        return {
            "partition": partition,
            "num_communities": len(set(partition.values())),
            "statistics": community_insights,
        }

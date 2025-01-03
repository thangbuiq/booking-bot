from abc import ABC, abstractmethod

from neo4j import GraphDatabase


class BaseHotelRecommender(ABC):
    def __init__(self, uri: str, username: str, password: str):
        """
        Base class for hotel recommendation systems.

        Args:
            uri (str): URI for the Neo4j database.
            username (str): Username for the Neo4j database.
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        """
        Close the connection to the Neo4j database.
        """
        self.driver.close()

    @abstractmethod
    def _create_constraints(self) -> None:
        """
        Create constraints in the Neo4j database.
        """
        raise NotImplementedError

    @abstractmethod
    def recommend_hotels(self) -> None:
        """
        Recommend hotels to users.
        """
        raise NotImplementedError

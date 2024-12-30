import os
from typing import Any
from typing import Dict
from typing import List

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


class HotelGraphDB:
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize the connection to the Neo4j database.

        Args:
            uri (str): URI of the Neo4j database.
            username (str): Username to connect to the database.
            password (str): Password to connect to the database.
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        """
        Close the connection to the Neo4j database.
        """
        self.driver.close()

    def _create_constraints(self) -> None:
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:StayType) REQUIRE s.name IS UNIQUE",
                # "CREATE CONSTRAINT IF NOT EXISTS FOR (st:StayDuration) REQUIRE st.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
                # "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Amenity) REQUIRE a.name IS UNIQUE",
            ]
            for constraint in constraints:
                session.run(constraint)

    def load_data(
        self,
        fact_review: pd.DataFrame,
        dim_hotel: pd.DataFrame,
        dim_user: pd.DataFrame,
        batch_size: int = 1000,
    ) -> None:
        """
        Load the data into the Neo4j database.

        Args:
            fact_review (pd.DataFrame): Fact table containing the reviews.
            dim_hotel (pd.DataFrame): Dimension table containing the hotels.
            dim_user (pd.DataFrame): Dimension table containing the users.
            batch_size (int): Number of records to process in each batch. Defaults to 1000.
        """
        self._create_constraints()

        with self.driver.session() as session:
            # Create countries

            #  Create locations
            locations_query = """
            UNWIND $locations AS location
            MERGE (l:Location {name: location})
            """
            session.run(locations_query, locations=dim_hotel["location"].unique())

            # Load hotels with relationships
            hotels_query = """
            UNWIND $hotels AS hotel
            MERGE (h:Hotel {hotel_id: hotel.hotel_id})
            SET h.name = hotel.name_hotel,
                h.address = hotel.address,
                h.description = hotel.descriptions,
                h.url = hotel.url_hotel
            WITH h, hotel
            MATCH (l:Location {name: hotel.location})
            MERGE (h)-[:LOCATED_IN]->(l)
            """

            for i in range(0, len(dim_hotel), batch_size):
                batch = dim_hotel.iloc[i : i + batch_size]
                session.run(hotels_query, hotels=batch.to_dict("records"))

            # Create stay types
            stay_types_query = """
            UNWIND $types AS type
            MERGE (s:StayType {name: type})
            """
            stay_types = fact_review["stay_type"].dropna().unique()
            session.run(stay_types_query, types=stay_types)

            # Load users with relationships
            users_query = """
            UNWIND $users AS user
            MERGE (u:User {user_id: user.user_id})
            SET u.username = user.username
            """

            for i in range(0, len(dim_user), batch_size):
                batch = dim_user.iloc[i : i + batch_size]
                session.run(users_query, users=batch.to_dict("records"))

            # Load reviews as relationships
            reviews_query = """
            UNWIND $reviews AS review
            MATCH (u:User {user_id: review.user_id})
            MATCH (h:Hotel {hotel_id: review.hotel_id})
            MATCH (s:StayType {name: review.stay_type})
            MERGE (u)-[r:REVIEWED {review_id: review.review_id}]->(h)
            SET r.rating = review.review_rating,
                r.date = datetime(review.review_post_date),
                r.stay_duration = review.stay_duration,
                r.title = review.review_title,
                r.liked = review.review_text_liked,
                r.disliked = review.review_text_disliked,
                r.room_view = review.room_view,
                r.timestamp = datetime(review.review_post_date).epochSeconds,
                r.stay_type = review.stay_type
            """

            for i in range(0, len(fact_review), batch_size):
                batch = fact_review.iloc[i : i + batch_size]
                session.run(reviews_query, reviews=batch.to_dict("records"))

            # Create similarity relationships
            similarity_query = """
            MATCH (h1:Hotel)<-[r1:REVIEWED]-(u:User)-[r2:REVIEWED]->(h2:Hotel)
            WHERE h1 <> h2 
            WITH h1, h2, count(DISTINCT u) as common_users,
                 avg(abs(r1.rating - r2.rating)) as rating_diff
            WHERE common_users >= 3
            MERGE (h1)-[s:SIMILAR_TO]-(h2)
            SET s.score = common_users / (1 + rating_diff)
            """
            session.run(similarity_query)

    def query(self, query: str, **kwargs) -> List[Dict[Any, Any]]:
        with self.driver.session() as session:
            results = session.run(query, **kwargs)
            return [dict(record) for record in results]

    def get_hotel_recommendations(
        self, user_id: str, limit: int = 5, min_rating: float = 4.0
    ) -> List[Dict[Any, Any]]:
        """
        Get hotel recommendations for a given user.

        Args:
            user_id (str): User ID for which to get recommendations.
            limit (int): Number of recommendations to return. Defaults to 5.
            min_rating (float): Minimum rating for the recommended hotels. Defaults to 4.0.

        Returns:
            List[Dict[Any, Any]]: List of recommended hotels with details.
        """
        query = """
        MATCH (target_user:User {user_id: $user_id})-[past_review:REVIEWED]->(past_hotel:Hotel)
        WITH target_user, 
            avg(coalesce(past_review.rating, 0)) as user_avg_rating,  // Use coalesce to handle null ratings
            collect(past_hotel) as visited_hotels

        MATCH (other_user:User)-[r1:REVIEWED]->(candidate:Hotel)
        WHERE other_user <> target_user 
        AND NOT candidate IN visited_hotels
        AND coalesce(r1.rating, 0) >= $min_rating  // Handle null ratings explicitly

        OPTIONAL MATCH (past_hotel)-[sim:SIMILAR_TO]-(candidate)

        WITH candidate,
            count(DISTINCT r1) as review_count,
            avg(coalesce(r1.rating, 0)) as avg_rating,  // Handle null ratings
            collect(DISTINCT coalesce(r1.stay_type, 'Unknown')) as stay_types,  // Replace null stay types with 'Unknown'
            coalesce(avg(sim.score), 0) as similarity_score  // Handle null similarity scores

        RETURN candidate.hotel_id as hotel_id,
            candidate.name as name,
            avg_rating,
            review_count,
            stay_types,
            (avg_rating * 0.4 + 
                log(review_count) * 0.2 +
                similarity_score * 0.4) as recommendation_score
        ORDER BY recommendation_score DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            results = session.run(
                query, user_id=user_id, limit=limit, min_rating=min_rating
            )
            return [dict(record) for record in results]

    def get_similar_hotels(self, hotel_id: str, limit: int = 5) -> List[Dict[Any, Any]]:
        query = """
        MATCH (h:Hotel {hotel_id: $hotel_id})
        OPTIONAL MATCH (h)-[sim:SIMILAR_TO]-(similar:Hotel)
        WITH h, similar, sim.score as similarity_score
        WHERE similar IS NOT NULL
        MATCH (similar)<-[r:REVIEWED]-()
        WITH similar, similarity_score,
            avg(r.rating) as avg_rating,
            count(r) as review_count
        WHERE avg_rating >= 8.0  // Only include highly rated hotels
        RETURN similar.hotel_id as hotel_id,
            similar.name as name,
            avg_rating,
            review_count,
            similarity_score,
            (similarity_score * 0.6 + avg_rating * 0.4) as recommendation_score
        ORDER BY recommendation_score DESC
        LIMIT $limit
        """

        fallback_query = """
        MATCH (h:Hotel {hotel_id: $hotel_id})<-[r1:REVIEWED]-()
        WITH h, avg(r1.rating) as target_rating
        MATCH (other:Hotel)<-[r2:REVIEWED]-()
        WHERE other <> h
        WITH other, target_rating, avg(r2.rating) as avg_rating, count(r2) as review_count
        WHERE avg_rating >= target_rating - 0.5
        RETURN other.hotel_id as hotel_id,
            other.name as name,
            avg_rating,
            review_count,
            0 as similarity_score,
            avg_rating as recommendation_score
        ORDER BY recommendation_score DESC, review_count DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            results = session.run(query, hotel_id=hotel_id, limit=limit)
            recommendations = [dict(record) for record in results]

            if not recommendations:
                results = session.run(fallback_query, hotel_id=hotel_id, limit=limit)
                recommendations = [dict(record) for record in results]

            return recommendations


if __name__ == "__main__":
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    graph_db = HotelGraphDB(
        os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
    )

    base_df_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "scraper",
        "booking",
        "output",
        "warehouse",
    )

    fact_review_df = pd.read_parquet(os.path.join(base_df_path, "fact_review.parquet"))
    dim_hotel_df = pd.read_parquet(os.path.join(base_df_path, "dim_hotel.parquet"))
    dim_user_df = pd.read_parquet(os.path.join(base_df_path, "dim_user.parquet"))

    graph_db.load_data(fact_review_df, dim_hotel_df, dim_user_df)

    test_user, test_hotel = (
        "d49aedca2af303f3439c9ddcfaa6c534",
        "d25d0d6a3f0f42b19482c3d6f16b7fbf",
    )
    print("Recommendations for user:", test_user)
    print(graph_db.get_hotel_recommendations(test_user))

    print("\nSimilar hotels to:", test_hotel)
    print(graph_db.get_similar_hotels(test_hotel))

    print("\nSample query:")
    query = """
    MATCH (h:Hotel)
    RETURN h.hotel_id as hotel_id, h.name as name
    LIMIT 5
    """

    print(graph_db.query(query))
